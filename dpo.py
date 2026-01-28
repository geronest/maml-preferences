import os, json
import argparse
from typing import List
import time

import torch

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    set_seed,
    Gemma3ForCausalLM
)

from trl import (
    DPOConfig, 
    DPOTrainer
)
from utils import Prompter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rl_training
from dataset_util import create_datasets_metadpo
from maml_trainer import MAMLTrainer
from maml_dataset import MAMLDataset


def train_model(args):
    if "gemma" in args.model_path:
        model_type = Gemma3ForCausalLM 
    else:
        model_type = AutoModelForCausalLM
    if args.use_lora:
        model, tokenizer = rl_training.load_quantized_lora_model(model_type, args.model_path)
    else:
        model, tokenizer = rl_training.load_model(model_type, args.model_path)

    print("Load DPO model")
    train_dataset, valid_dataset, test_dataset = create_datasets_metadpo(args, tokenizer)
    print("Load DPO dataset")

    maml_train_dataset = MAMLDataset(
        train_dataset, 
        args.inner_train_batch_size*2, # split inside the trainer for batch_inner and batch_outer
        args.num_tasks_per_batch,
        steps_per_epoch=args.max_steps
    )
    maml_val_dataset = MAMLDataset(
        valid_dataset, 
        args.eval_batch_size,
        args.num_tasks_per_batch,
        steps_per_epoch=args.eval_steps,
        sample_all = True
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,
        max_steps = args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,
        save_total_limit=2,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        # per_device_train_batch_size=args.batch_size,
        # per_device_eval_batch_size=args.eval_batch_size,

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        # local_rank=args.local_rank,
        label_names=[],
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        max_prompt_length=128,
        max_completion_length=512,


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


    trainer = MAMLTrainer(
        model_type="dpo",
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
    # trainer.evaluate()
    trainer.train()
    print("Finish training DPO model")

language_list = ['ro', 'it', 'es', 'ca']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpts/rlhf/")
    parser.add_argument("--data_path", type=str, default="datasets/multilingual-ranking-data-42k/ar.json")
    parser.add_argument("--language_list", type=str, nargs="+", default=language_list, help='Split by space')
    
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--seq_length", type=int, default=512)

    parser.add_argument("--num_epochs", type=int, default=1)
    # parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="ckpts/adaption_dpo/policy_ar")
    parser.add_argument("--log_freq", default=10, type=int)
    
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--wandb_name", default='RL-training', type=str)

    parser.add_argument("--eval_batch_size", type=int, default=4) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inner_train_batch_size", default=2, type=int)
    parser.add_argument("--inner_optimizer", default='SGD', type=str)
    parser.add_argument("--inner_lr_coef", default=3, type=float)
    parser.add_argument("--num_tasks_per_batch", default=2, type=int)
    parser.add_argument("--use_same_languages_for_eval", action="store_true")
    parser.add_argument("--type_model", type=str, default="dpo",
                        help="type of the model. it is fixed to reward so no need to change")

 
    parser.add_argument("--use_lora", action="store_true")
    # parser.add_argument("--micro_batch_size", type=int, default=4)

    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--train_dataset_size", default=None, type=int)

    args = parser.parse_args()

    set_seed(args.seed)

    start_time = time.time()
    train_model(args)
    end_time = time.time()

    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")
