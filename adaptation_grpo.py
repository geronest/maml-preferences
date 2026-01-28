import os, json
import argparse
from typing import List
import time

import torch

from datasets import load_dataset, Dataset, concatenate_datasets
from dataset_util import create_datasets_grpo
from transformers import (
    AutoModelForSequenceClassification,
    set_seed,
    TrainingArguments,
    Gemma3ForCausalLM
)

from trl import (
    GRPOConfig, 
    GRPOTrainer,
    RewardTrainer,
    RewardConfig,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rl_training, meta_supervised_finetuning

from utils import Prompter
prompter = Prompter()

def read_json_lora(args, type_model="policy"):
    target_check = args.policy_use_lora if type_model == "policy" else args.reward_use_lora
    target_path = args.policy_path if type_model == "policy" else args.reward_path
    if target_check:
        with open(target_path + "/adapter_config.json", "r") as fp:
            res_json = json.load(fp)
    else:
        raise NotImplementedError

    return res_json

def read_json_base(args, type_model="policy"):
    target_path = args.policy_path if type_model == "policy" else args.reward_path
    with open(target_path + "/config.json", "r") as fp:
        res_json = json.load(fp)

    return res_json

def policy_adaption(args):
    print("Create reward function")
    reward_fn = rl_training.create_reward_fn(args)

    # train_dataset, eval_dataset = create_datasets(args)
    train_dataset, valid_dataset, test_dataset = create_datasets_grpo(args)
    print("Create datasets")

    training_args = GRPOConfig(
        output_dir=args.policy_output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,

        max_steps = args.max_steps,
        eval_strategy="steps",
        eval_accumulation_steps=args.eval_freq,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_freq,
        save_total_limit=2,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_generations=args.num_generations,

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        local_rank=args.local_rank,
        label_names=[],
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        max_prompt_length=args.prompt_length,
        max_completion_length=args.seq_length,

        logging_dir="./logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.log_freq,

        report_to='wandb',
        run_name=args.wandb_name,
        no_cuda=False,
        remove_unused_columns=False,
    )
    training_args = training_args.set_dataloader(train_batch_size=args.batch_size, eval_batch_size=args.eval_batch_size)

    if args.policy_use_lora:
        policy_json = read_json_lora(args, "policy")
        if "gemma" in policy_json["base_model_name_or_path"]:
            model_type = Gemma3ForCausalLM
        else:
            model_type = rl_training.BloomWithLogitsToKeep
        print("#####\n    Using LoRA\n#####\n")
        policy_model, tokenizer = rl_training.load_quantized_lora_model(model_type, args.policy_path)
    else:
        policy_json = read_json_base(args, "policy")
        if "gemma" in policy_json["architectures"][0]:
            model_type = Gemma3ForCausalLM
        else:
            model_type = rl_training.BloomWithLogitsToKeep
        policy_model, tokenizer = rl_training.load_model(model_type, args.policy_path)
    
    print("Load policy model")
    trainer = GRPOTrainer(
        model=policy_model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.evaluate()
    trainer.train()
    print("Finish training policy model")

def get_args():
    """Parse command line arguments and return configuration objects."""
    parser = argparse.ArgumentParser()
    
    # Common arguments
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--data_path", type=str, default="datasets/multilingual-ranking-data-42k/")
    # parser.add_argument("--output_dir", type=str, default="ckpts/adaption_reward/")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--size_valid_set", type=float, default=0.05)
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--train_dataset_size", default=None, type=int)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--wandb_name", default='adaptation-grpo', type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=5)
    parser.add_argument("--reward_use_lora", action="store_true")
    parser.add_argument("--policy_use_lora", action="store_true")
    # parser.add_argument("--use_lora", action="store_true")

    parser.add_argument("--reward_path", type=str, default="ckpts/reward_models/checkpoint-500")
    
    parser.add_argument("--policy_path", type=str, default="ckpts/rlhf/")
    parser.add_argument("--policy_output_dir", type=str, default="ckpts/adaption_reward/policy")
    parser.add_argument("--policy_data_path", type=str, default="datasets/multilingual-rl-tuning-64k/")


    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    policy_args = get_args()

    set_seed(policy_args.seed)

    print("Training policy model")
    policy_adaption(policy_args)

    
