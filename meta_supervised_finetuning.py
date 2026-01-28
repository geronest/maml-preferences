import argparse
import os
import random

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from dataset_util import create_datasets_metasft

# from torch.utils.data import IterableDataset
# from tqdm import tqdm
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    cast_mixed_precision_params,
    get_peft_model,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    # DataCollatorForSeq2Seq,
    # Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    Gemma3TextConfig
)

from utils import Prompter
# from sft_trainer import SFTTrainer
from maml_trainer import MAMLTrainer
from maml_dataset import MAMLDataset
import rl_training

# All available languages
# 'sv' would cause error - may exist invalid data
language_list = ['ar', 'bn', 'ca', 'da', 'de', 'en', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'it', 'jp', 'kn', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'ta', 'te', 'uk', 'vi', 'zh']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_model", type=str, default="sft")
    parser.add_argument("--model_path", type=str, default="bloom-7b1", help="LLaMa weights that is converted to huggingface format!")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default='BloomBlock')

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--log_freq", default=10, type=int)

    parser.add_argument("--num_tasks_per_batch", default=5, type=int)
    parser.add_argument("--inner_train_batch_size", default=4, type=int)
    # parser.add_argument("--inner_lr", default=5e-7, type=float)
    parser.add_argument("--inner_lr_coef", default=3, type=float)
    parser.add_argument("--num_inner_steps", default=0, type=int)
    parser.add_argument("--name_optimizer", default='SGD', type=str)
    parser.add_argument("--wandb_name", default='Meta-SFT', type=str)
    parser.add_argument("--language_list", type=str, nargs="+", default=language_list, help='Split by space')

    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_results_path", type=str, default="./eval_results/")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_same_languages_for_eval", action="store_true")

    return parser.parse_args()

def save_eval_results(args, eval_results):
    os.makedirs(args.eval_results_path, exist_ok=True)

    # Combine args and eval_results into a single row
    row_dict = vars(args)
    row_dict.update(eval_results)

    # Create DataFrame with single row
    df_new = pd.DataFrame([row_dict])

    # Check if file exists and load it if it does
    results_file = os.path.join(args.eval_results_path, "results.csv")
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    # Save updated DataFrame
    df.to_csv(results_file, index=False)

def load_model(args, tokenizer):
    print('Loading model...')
    
    if args.use_lora:
        # Quantization + LoRA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        modelType = Gemma3ForCausalLM if 'gemma' in args.model_path else AutoModelForCausalLM
        model = modelType.from_pretrained(
                args.model_path,
                # trust_remote_code=True,
                device_map=device,
                low_cpu_mem_usage=True,
                # use_cache=False,
                quantization_config=quantization_config,
            )
        print("Model initialized: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        model.config.pad_token_id = tokenizer.pad_token_id
        model = prepare_model_for_kbit_training(model)

        if 'bloom' in args.model_path:
            modules =  ["query_key_value"]
        elif "Apertus" in args.model_path:
            modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            modules = ['gate_proj', 'down_proj', 'v_proj', 'k_proj', 'q_proj', 'o_proj', 'up_proj']
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )
    
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
        cast_mixed_precision_params(model, dtype=torch.float16)
        model.print_trainable_parameters()
    else:
        modelType = Gemma3ForCausalLM if 'gemma' in args.model_path else AutoModelForCausalLM
        model = modelType.from_pretrained(
            args.model_path,
            # trust_remote_code=True,
            device_map=device,
            low_cpu_mem_usage=True,
            # use_cache=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        print("Model initialized: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model

def run_training(args, train_dataset, eval_dataset, tokenizer):
    model = load_model(args, tokenizer)
    
    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,

        max_steps = args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,

        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        # optim="adamw_torch",
        learning_rate=args.learning_rate,
        # adam_beta1=0.9,
        # adam_beta2=0.95,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,

        logging_dir="./logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.log_freq,

        report_to='wandb', #wandb
        run_name=args.wandb_name,
        no_cuda=False,
        remove_unused_columns=False,
        # fsdp="full_shard auto_wrap",
        # fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
    )
    training_args = training_args.set_dataloader(train_batch_size=args.micro_batch_size, eval_batch_size=args.eval_batch_size)

    maml_train_dataset = MAMLDataset(
        train_dataset, 
        args.inner_train_batch_size*2,
        args.num_tasks_per_batch,
        steps_per_epoch=args.max_steps
    )
    maml_eval_dataset = MAMLDataset(
        eval_dataset, 
        args.eval_batch_size,
        args.num_tasks_per_batch,
        steps_per_epoch=args.eval_steps,
        sample_all = True
    )

    trainer = MAMLTrainer(
        model=model,
        args=training_args,
        train_dataset=maml_train_dataset,
        eval_dataset=maml_eval_dataset,
        processing_class=tokenizer,
        inner_train_batch_size=args.inner_train_batch_size,
        inner_lr_coef=args.inner_lr_coef,
        inner_optimizer=args.name_optimizer,
        num_inner_steps=args.num_inner_steps,
    )

    # model.config.use_cache = False

    # test_model(model, maml_train_dataset)

    print("Training...")
    trainer.train()
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(args.output_dir, state_dict=cpu_state_dict)  # noqa
        tokenizer.save_pretrained(args.output_dir)

    print("Evaluating...")
    eval_results = trainer.evaluate()
    save_eval_results(args, eval_results)
    print("Training Done!")

def test_model(model, dataset):
    from torch.func import functional_call
    params_func = {n: p.detach().clone().requires_grad_() for n, p in model.named_parameters() if p.requires_grad}
    for n in params_func:
        print(f"[params_func_check] {n}: {params_func[n].requires_grad}")
    
    opt2 = torch.optim.SGD(params_func.values(), lr=1e-4)

    inputs = dataset[0]
    inputs_focus = inputs[list(inputs.keys())[0]]
    inputs_focus = {k: inputs_focus[k].to(model.device) for k in inputs_focus}

    params_name = list(params_func.keys())
    
    opt2.zero_grad()
    # Generate outputs
    outputs = functional_call(model, params_func, (), kwargs=inputs_focus)

    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    loss.backward()
    opt2.step()

def main(args):
    print('Start config')
    configType = Gemma3TextConfig if 'gemma' in args.model_path else AutoConfig
    config = configType.from_pretrained(args.model_path)
    architecture = config.architectures[0]
    print('Start tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print('End tokenizer')

    if "Llama" in architecture:
        print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }
        )
        tokenizer.pad_token_id = (
            0
        )
        args.fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"
    elif 'Bloom' in architecture:
        args.fsdp_transformer_layer_cls_to_wrap = "BloomBlock"
    elif 'Gemma' in architecture:
        args.fsdp_transformer_layer_cls_to_wrap = "Gemma3DecoderLayer"
    # else:
    #     raise ValueError("We only support Llama, Bloom, and Gemma models")

    # train_dataset, eval_dataset = create_datasets(tokenizer, args)
    train_dataset, valid_dataset, test_dataset = create_datasets_metasft(tokenizer, args)
    # print(train_dataset.shape)
    for lang, dataset in train_dataset.items():
        # print(lang)
        s_lang = f"[{lang}] {len(dataset)} items"
        for key, item in dataset[0].items():
            # print(key)
            # print(len(item))
            s_lang += f" | {key}: {len(item)}"
        print(s_lang)

    print("Dataset prepared: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    run_training(args, train_dataset, valid_dataset, tokenizer)

if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
