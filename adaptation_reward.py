import os, json
import argparse
from typing import List
import time

import torch

from datasets import load_dataset, Dataset, concatenate_datasets
from dataset_util import load_reward_dataset
from transformers import (
    AutoModelForSequenceClassification,
    set_seed,
    TrainingArguments,
)

from trl import (
    GRPOConfig, 
    GRPOTrainer,
    RewardTrainer,
    RewardConfig,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rl_training
from reward_modeling import load_model as create_new_reward_model

def train_reward_model(args):
    try:
        # reward_model, reward_tokenizer = rl_training.load_quantized_lora_model(AutoModelForSequenceClassification, args.reward_path)
        reward_model, reward_tokenizer = rl_training.load_quantized_reward_model(AutoModelForSequenceClassification, args.reward_path)
        print(f"########\n    Reward Model loaded from: {args.reward_path}\n########\n")
    except:
        args_reward_model = {
            "model_path": args.reward_path,
            "fsdp_transformer_layer_cls_to_wrap": "BloomBlock",
            "use_lora": args.use_lora
        }
        args_reward_model = argparse.Namespace(**args_reward_model)
        reward_model, reward_tokenizer = create_new_reward_model(args_reward_model)
        print(f"########\n    New Reward Model initialized: {args.reward_path}\n########\n")
    reward_train, reward_valid, reward_test = load_reward_dataset(args, reward_tokenizer)

    training_args = RewardConfig(
        output_dir=args.reward_output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,

        max_steps = args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,
        save_total_limit=2,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        local_rank=args.local_rank,
        label_names=[],
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
        bf16=args.bf16,
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=reward_train,
        eval_dataset=reward_valid,
        processing_class=reward_tokenizer,)
    
    trainer.evaluate()
    trainer.train()

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
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--train_dataset_size", default=None, type=int)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--wandb_name", default='adaptation-reward', type=str)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=5)
    parser.add_argument("--use_lora", action="store_true")

    # Reward model specific arguments
    parser.add_argument("--reward_path", type=str, default="ckpts/reward_models/checkpoint-500")
    parser.add_argument("--reward_output_dir", type=str, default="ckpts/adaption_reward/reward_model")
    parser.add_argument("--reward_data_path", type=str, default="datasets/multilingual-ranking-data-42k/")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    reward_args = get_args()

    set_seed(reward_args.seed)
    print("Training reward model")
    train_reward_model(reward_args)

    
