import os

import argparse
# from typing import Optional
# from dataclasses import dataclass
import random

import torch
from datasets import load_dataset, DatasetDict
from dataset_util import create_datasets_metareward
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    # Trainer,
    TrainingArguments,
    # PreTrainedTokenizerBase,
    set_seed,
    logging,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    Gemma3ForSequenceClassification,
    Gemma3TextConfig,
    Gemma3ForCausalLM
)

from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    cast_mixed_precision_params,
    get_peft_model,
    PeftModel,
)

from torch.func import functional_call
import torch.nn as nn


from utils import Prompter
from maml_trainer import MAMLTrainer
from maml_dataset import MAMLDataset
from other_peft_model_to_sequence_classification import ConvertedPeftModelForSequenceClassification


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# multilingual-ranking-data-42k/ does not contain "en", "jp", "sv"
# language_list = ['ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'it', 'kn', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'ta', 'te', 'uk', 'vi', 'zh']
language_list = ['ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'it', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'ta', 'te', 'uk', 'vi', 'zh']

def load_full_model(args):
    modelType = Gemma3ForSequenceClassification if 'gemma' in args.model_path else AutoModelForSequenceClassification
    model = modelType.from_pretrained(
        args.model_path,
        # trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
        num_labels=1,
        # use_cache=False,
        torch_dtype=torch.bfloat16
    )

    return model

def load_lora_model(args):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if 'gemma' in args.model_path or 'apertus' in args.model_path:
        model = load_lora_customized(args, quantization_config)
    else:
        model = load_lora_no_gemma(args, quantization_config)
    return model

def load_lora_no_gemma(args, quantization_config):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        # trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
        num_labels=1,
        # use_cache=False,
        quantization_config=quantization_config,
        )
    # accelerator.print("Model initialized: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        inference_mode=False,
        target_modules=["query_key_value"],
    )

    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    # cast_mixed_precision_params(model, dtype=torch.float16)
    model.print_trainable_parameters()
    return model

def load_lora_customized(args, quantization_config):
    model_type = Gemma3ForCausalLM if 'gemma' in args.model_path else AutoModelForCausalLM
    model = model_type.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16, 
        device_map=device,
        attn_implementation='eager',
        quantization_config=quantization_config
        )
    model.lm_head = torch.nn.Linear(model.config.hidden_size, 1, bias=False,device=device)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = ['gate_proj', 'down_proj', 'v_proj', 'k_proj', 'q_proj', 'o_proj', 'up_proj']
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.__class__ = ConvertedPeftModelForSequenceClassification
    model.num_labels = 1

    return model

def load_model(args):
    print('Loading model...')
    configType = Gemma3TextConfig if 'gemma' in args.model_path else AutoConfig
    config = configType.from_pretrained(args.model_path)
    architecture = config.architectures[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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
    #     raise ValueError("We only support Llama and Bloom models")

    model = load_lora_model(args) if args.use_lora else load_full_model(args)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def run_training(args):
    print("Start creating model and tokenizer")
    model, tokenizer = load_model(args)
    print("Start creating datasets")
    # train_data, val_data = create_datasets(args, tokenizer)
    train_data, val_data, test_data = create_datasets_metareward(args, tokenizer)
    
    maml_train_dataset = MAMLDataset(
        train_data, 
        args.inner_train_batch_size*2, # split inside the trainer for batch_inner and batch_outer
        args.num_tasks_per_batch,
        steps_per_epoch=args.max_steps
    )
    maml_val_dataset = MAMLDataset(
        val_data, 
        args.eval_batch_size,
        args.num_tasks_per_batch,
        steps_per_epoch=args.eval_steps,
        sample_all = True
    )
    print("All set")


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

        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_total_limit=2,
        # save_total_limit=args.save_limit if args.save_limit is not None else None,

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # fp16=not args.bf16,
        bf16=args.bf16,
        local_rank=args.local_rank,
        label_names=[],
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,

        report_to='wandb',
        run_name=args.wandb_name,
        no_cuda=False,
        remove_unused_columns=False,

        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False, 
    )
    training_args = training_args.set_dataloader(train_batch_size=args.micro_batch_size, eval_batch_size=args.eval_batch_size)
    
    trainer = MAMLTrainer(
        model_type="reward",
        model=model,
        args=training_args,
        train_dataset=maml_train_dataset,
        eval_dataset=maml_val_dataset,
        processing_class=tokenizer,
        inner_train_batch_size=args.inner_train_batch_size,
        inner_lr_coef=args.inner_lr_coef,
        inner_optimizer=args.inner_optimizer,
        # callbacks=[early_stopping_callback]
        )
    trainer.train()
    print("Saving last checkpoint of the model")

    # best_model_dir = os.path.join(training_args.output_dir, "best_model")
    # trainer.save_model(best_model_dir)

    if trainer.args.should_save:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(args.output_dir, state_dict=cpu_state_dict)  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="bigscience/bloom-7b1")
    parser.add_argument("--model_path", type=str, default="bigscience/bloom-7b1",
                        help="LLaMa/Bloom weights that is converted to huggingface format!")
    parser.add_argument("--type_model", type=str, default="reward",
                        help="type of the model. it is fixed to reward so no need to change")
    parser.add_argument("--data_path", type=str, default="dataset/moose-ranking/vi.json")

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--seq_length", type=int, default=768)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/reward_model")

    parser.add_argument("--num_tasks_per_batch", default=2, type=int)
    parser.add_argument("--inner_train_batch_size", default=2, type=int)
    # parser.add_argument("--inner_lr", default=3e-5, type=float)
    parser.add_argument("--inner_lr_coef", default=3, type=float)
    parser.add_argument("--inner_optimizer", default='SGD', type=str)
    parser.add_argument("--wandb_name", default='reward-modeling', type=str)
    parser.add_argument("--language_list", type=str, nargs="+", default=language_list, help='Split by space')
    
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_freq", default=300, type=int)
    parser.add_argument("--save_freq", default=300, type=int)
    parser.add_argument("--log_freq", default=20, type=int)

    parser.add_argument("--eval_batch_size", type=int, default=2) 
    parser.add_argument("--micro_batch_size", type=int, default=2) # Batch size of tasks in training

    parser.add_argument("--save_limit", type=int, default=None)
    parser.add_argument("--train_strategy", type=str, default='steps')
    parser.add_argument("--use_same_languages_for_eval", action="store_true")
    parser.add_argument("--use_lora", action="store_true")

    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    # args.gradient_accumulation_steps = args.batch_size // (args.micro_batch_size*n_gpus)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    run_training(args)
