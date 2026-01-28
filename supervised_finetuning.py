import argparse
import os
import json

import pandas as pd
import torch
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    cast_mixed_precision_params,
    get_peft_model,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    ApertusForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)

from sft_trainer import SFTTrainer
from dataset_util import create_datasets_sft
import rl_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_json_lora(checkpoint_path):
    """Read LoRA adapter config from checkpoint."""
    with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as fp:
        return json.load(fp)


def read_json_base(checkpoint_path):
    """Read base model config from checkpoint."""
    with open(os.path.join(checkpoint_path, "config.json"), "r") as fp:
        return json.load(fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bloom-7b1", help="Model weights path")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--train_dataset_size", type=int, default=100)
    parser.add_argument("--eval_dataset_size", type=int, default=1000)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default='BloomBlock')

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--log_freq", default=10, type=int)

    parser.add_argument("--wandb_name", default='SFT', type=str)

    parser.add_argument("--max_steps", type=int, default=2500)
    # parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_results_path", type=str, default="./eval_results/")
    parser.add_argument("--use_lora", action="store_true")

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


def run_training(args, train_dataset, eval_dataset, tokenizer):

    # Auto-detect if model_path is a checkpoint
    is_checkpoint = False
    is_lora_checkpoint = False
    
    # Check if it's a LoRA checkpoint
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        is_checkpoint = True
        is_lora_checkpoint = True
        print(f"########\n    Detected LoRA checkpoint at: {args.model_path}\n########\n")
    # Check if it's a full model checkpoint (has both config.json and other training artifacts)
    elif os.path.exists(os.path.join(args.model_path, "config.json")) and \
         (os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")) or \
          os.path.exists(os.path.join(args.model_path, "model.safetensors"))):
        is_checkpoint = True
        is_lora_checkpoint = False
        print(f"########\n    Detected full model checkpoint at: {args.model_path}\n########\n")
    
    if is_checkpoint:
        # Load from checkpoint
        if is_lora_checkpoint:
            checkpoint_json = read_json_lora(args.model_path)
            base_model_name = checkpoint_json["base_model_name_or_path"]
            
            # Determine model type
            if "Apertus" in base_model_name:
                model_type = ApertusForCausalLM
            else:
                model_type = AutoModelForCausalLM
            
            model, _ = rl_training.load_quantized_lora_model(model_type, args.model_path)
        else:
            checkpoint_json = read_json_base(args.model_path)
            
            # Determine model type
            if "Apertus" in checkpoint_json.get("architectures", [""])[0]:
                model_type = ApertusForCausalLM
            else:
                model_type = AutoModelForCausalLM
            
            model, _ = rl_training.load_model(model_type, args.model_path)
        
        print("Model loaded from checkpoint")
    else:
        # Original logic: create new model from scratch
        print(f"########\n    Creating new model from base: {args.model_path}\n########\n")
        if "Apertus" in args.model_path:
            causal_lm = ApertusForCausalLM
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            causal_lm = AutoModelForCausalLM
            target_modules=["query_key_value"]

        if args.use_lora:
            # Quantization + LoRA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = causal_lm.from_pretrained(
                args.model_path,
                device_map=device,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
            )
            print("Model initialized: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            model.config.pad_token_id = tokenizer.pad_token_id
            model = prepare_model_for_kbit_training(model)
        
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
        
            model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
            cast_mixed_precision_params(model, dtype=torch.float16)
        else:
            model = causal_lm.from_pretrained(
                args.model_path,
                device_map=device,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16
            )
            print("Model initialized: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            model.config.pad_token_id = tokenizer.pad_token_id

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,

        max_steps=args.max_steps,

        eval_strategy="steps",
        eval_steps=args.eval_freq,

        save_strategy="steps",
        save_steps=args.save_freq,

        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        optim="adamw_torch",
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,

        logging_dir="./logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.log_freq,

        report_to='wandb',
        run_name=args.wandb_name,
        no_cuda=False,
        remove_unused_columns=False,
    )
    training_args = training_args.set_dataloader(train_batch_size=args.micro_batch_size, eval_batch_size=args.eval_batch_size)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(args.output_dir)

    print("Evaluating...")
    eval_results = trainer.evaluate()
    save_eval_results(args, eval_results)
    print("Training Done!")


def main(args):
    print('Start config')
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Check if model_path is a LoRA checkpoint
    is_lora_checkpoint = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    if args.use_lora and is_lora_checkpoint:
        # Loading from existing LoRA checkpoint
        checkpoint_json = read_json_lora(args.model_path)
        base_model_name = checkpoint_json["base_model_name_or_path"]
        if "Apertus" in base_model_name:
            architecture = "ApertusForCausalLM"
        else:
            architecture = "AutoModelForCausalLM"
    else:
        # Creating new model or loading full model checkpoint
        architecture = config.architectures[0]
        base_model_name = args.model_path
    
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
        tokenizer.pad_token_id = 0
        args.fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"
    elif 'Bloom' in architecture or "bloom" in base_model_name:
        args.fsdp_transformer_layer_cls_to_wrap = "BloomBlock"
    elif "Apertus" in architecture:
        args.fsdp_transformer_layer_cls_to_wrap = "ApertusDecoderLayer"
    else:
        raise ValueError("We only support Llama, Bloom, and Apertus models")

    train_dataset, eval_dataset, _ = create_datasets_sft(tokenizer, args)
    print("Dataset prepared: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    run_training(args, train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
