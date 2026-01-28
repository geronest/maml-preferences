import os, json
import argparse
from typing import List
import time

import torch

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForCausalLM
    set_seed,
)

from trl import (
    DPOConfig, 
    DPOTrainer
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rl_training, adaptation_reward

from utils import Prompter
prompter = Prompter()

def create_datasets(args):
    def preprocess(data_point):
        prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
        )

        return {
            "prompt": prompt,
            "chosen": data_point["prefered_output"],
            "rejected": data_point["rejected_output"],
        }
    
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

    initial_chosen_columns = ['instruction','input', 'prefered_output', 'rejected_output']
    dataset = dataset.select_columns(initial_chosen_columns)

    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)

    if args.train_dataset_size is not None:
        train_prompts = dataset["train"].shuffle(seed=args.seed).map(preprocess).select(list(range(args.train_dataset_size)))
    else:
        train_prompts = dataset["train"].shuffle(seed=args.seed).map(preprocess)
    
    valid_prompts = dataset["test"].map(preprocess).select(list(range(args.eval_dataset_size)))

    train_prompts = train_prompts.remove_columns(initial_chosen_columns)
    valid_prompts = valid_prompts.remove_columns(initial_chosen_columns)

    print(f"Size of the train set: {len(train_prompts)}. Size of the validation set: {len(valid_prompts)}")

    return train_prompts, valid_prompts


def train_model(args):
    if args.use_lora:
        model, tokenizer = rl_training.load_quantized_lora_model(AutoModelForCausalLM, args.model_path)
    else:
        model, tokenizer = rl_training.load_model(AutoModelForCausalLM, args.model_path)
    print("Load DPO model")
    train_dataset, eval_dataset = create_datasets(args)
    print("Load DPO dataset")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,
        beta=args.beta,

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
        bf16=args.bf16,
        local_rank=args.local_rank,
        label_names=[],
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        gradient_checkpointing = True,
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


    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,)

    trainer.evaluate()
    trainer.train()
    print("Finish training DPO model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpts/rlhf/")
    parser.add_argument("--data_path", type=str, default="datasets/multilingual-ranking-data-42k/ar.json")

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="ckpts/adaption_dpo/policy_ar")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--wandb_name", default='adaptation-dpo', type=str)
    
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_freq", default=500, type=int)

    parser.add_argument("--eval_batch_size", type=int, default=4) 
    parser.add_argument("--batch_size", type=int, default=4) 

    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--train_dataset_size", default=None, type=int)
    parser.add_argument("--use_lora", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    start_time = time.time()
    train_model(args)
    end_time = time.time()

    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")
