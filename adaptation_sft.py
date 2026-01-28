import argparse
import os, json
import random

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk

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
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    Gemma3TextConfig
)
import shutil

# from sft_trainer import SFTTrainer
from utils import Prompter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bloom-7b1", help="LLaMa weights that is converted to huggingface format!")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default='BloomBlock')

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--name_optimizer", default='SGD', type=str)
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=300, type=int)
    parser.add_argument("--log_freq", default=10, type=int)
    

    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--train_dataset_size", default=None, type=int)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_results_path", type=str, default="./eval_results/")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--wandb_name", default='reward-modeling', type=str)

    parser.add_argument("--save_limit", type=int, default=None)
    parser.add_argument("--train_strategy", type=str, default='steps')

    return parser.parse_args()



def create_datasets(tokenizer, args):
    """Create the datasets for training and validation."""
    def tokenize(data_point, add_eos_token=True):
        prompt = [prompter.generate_prompt(instr, input, output)
                  for instr, input, output in
                  zip(data_point["instruction"], data_point["input"], data_point["output"])]

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.seq_length,
            padding='max_length',
            return_tensors='pt',
        )

        for i in range(len(result["input_ids"])):
            if (
                result["input_ids"][i][-1] != tokenizer.eos_token_id
                and len(result["input_ids"][i]) < args.seq_length
                and add_eos_token
            ):
                result["input_ids"][i].append(tokenizer.eos_token_id)
                result["attention_mask"][i].append(1)

        result["labels"] = result["input_ids"].clone()

        return result

    prompter = Prompter()
    try:
        dataset = load_dataset('json', split=args.split, data_files=args.data_path)
    except:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())
        for entry in dataset:
            for k, v in entry.items():
                if not isinstance(v, str):
                    entry[k] = str(v)
        dataset = Dataset.from_list(dataset)

    original_columns = dataset.column_names

    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)

    # accelerator.print(f"Training dataset limitation: {args.train_dataset_size}")
    print(f"Training dataset limitation: {args.train_dataset_size}")

    if args.train_dataset_size is not None:
        train_data = dataset["train"].shuffle().map(tokenize, batched=True, num_proc=64, remove_columns=original_columns).select(list(range(args.train_dataset_size)))
    else:
        train_data = dataset["train"].shuffle().map(tokenize, batched=True, num_proc=64, remove_columns=original_columns)
    valid_data = dataset["test"].shuffle(seed=args.seed).map(tokenize, batched=True, num_proc=8, remove_columns=original_columns).select(list(range(args.eval_dataset_size)))
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    train_data.set_format("torch")
    valid_data.set_format("torch")

    return train_data, valid_data

def load_full_model(model_type, model_path):
    print("Loading full policy model")
    model = model_type.from_pretrained(
        model_path,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    return model

def load_lora_model(model_type, model_path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = model_type.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map=device,
        quantization_config=quantization_config,
        attn_implementation="eager"
    )

    model = prepare_model_for_kbit_training(model)

    if 'bloom' in model_path:
        modules =  ["query_key_value"]
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

    return model



def run_training(args, train_dataset, eval_dataset, tokenizer):
    modelType = Gemma3ForCausalLM if 'gemma' in args.model_path else AutoModelForCausalLM
    model = load_lora_model(modelType, args.model_path) if args.use_lora else load_full_model(modelType, args.model_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,

        max_steps = args.max_steps if args.train_strategy == 'steps' else -1,

        eval_strategy=args.train_strategy,
        save_strategy=args.train_strategy,
        logging_strategy=args.train_strategy,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        logging_dir="./logs",
        logging_first_step=True,

        save_total_limit=args.save_limit if args.save_limit is not None else None,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        optim="adamw_torch",
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        bf16=True,
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,        

        report_to='wandb',
        run_name=args.wandb_name,
        no_cuda=False,
        remove_unused_columns=False,
    )
    training_args = training_args.set_dataloader(train_batch_size=args.batch_size, eval_batch_size=args.eval_batch_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.evaluate()
    trainer.train()

    print("Evaluating...")
    eval_results = trainer.evaluate()

    final_save_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_save_path)

    save_eval_results(args, eval_results)

    print("Training Done!")

def save_eval_results(args, eval_results):
    os.makedirs(args.eval_results_path, exist_ok=True)

    row_dict = vars(args)
    row_dict.update(eval_results)

    df_new = pd.DataFrame([row_dict])

    results_file = os.path.join(args.eval_results_path, "results.csv")
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    # Save updated DataFrame
    df.to_csv(results_file, index=False)


def main(args):
    print('Start config')
    # print(args.model_path)
    # accelerator.print('Start config')
    # configType = Gemma3TextConfig if 'gemma' in args.model_path else AutoConfig
    # config = configType.from_pretrained(args.model_path)
    # architecture = config.architectures[0]
    print('Start tokenizer')
    # accelerator.print('Start tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print('End tokenizer')
    # accelerator.print('End tokenizer')

    if "Llama" in args.model_path:
        print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        # accelerator.print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
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
    elif 'Bloom' in args.model_path:
        args.fsdp_transformer_layer_cls_to_wrap = "BloomBlock"
    elif 'Gemma' in args.model_path:
        args.fsdp_transformer_layer_cls_to_wrap = "Gemma3DecoderLayer"
    # else:
    #     raise ValueError("We only support Llama and Bloom models")

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    # for key, item in train_dataset[0].items():
    #     print(key)
    #     print(len(item))
    # print("Dataset prepared: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    run_training(args, train_dataset, eval_dataset, tokenizer)

if __name__ == "__main__":
    args = get_args()
    # accelerator = Accelerator()
    assert args.model_path != "", "Please provide the model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)


