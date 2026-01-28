import os, json
import argparse
from typing import List
import time
import pandas as pd

import torch
from tqdm import tqdm

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    ApertusForCausalLM,
    set_seed,
    TrainingArguments,
    AutoModelForCausalLM,
    Gemma3ForCausalLM
)

from trl import (
    GRPOConfig, 
    GRPOTrainer,
    RewardTrainer,
    RewardConfig,
)

from dataset_util import create_datasets_grpo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rl_training
from adaptation_grpo import read_json_lora, read_json_base


def generate_responses(args):
    print("Create reward function")
    reward_fn = rl_training.create_reward_fn(args)

    print("Create datasets")
    train_dataset, valid_dataset, test_dataset = create_datasets_grpo(args)

    if "gemma" in args.policy_path:
        model_type = Gemma3ForCausalLM 
    elif "bloom" in args.policy_path:
        model_type = rl_training.BloomWithLogitsToKeep
    else:
        model_type = AutoModelForCausalLM

    if args.policy_use_lora:
        print(f"#####\n    Using LoRA Policy Model: {args.policy_path}\n#####\n")
        policy_model, tokenizer = rl_training.load_quantized_lora_model(model_type, args.policy_path)
    else:
        policy_model, tokenizer = rl_training.load_model(model_type, args.policy_path)
    
    
    print("Load policy model")

    test_dataset = test_dataset[:args.eval_dataset_size]

    idx_start = 0

    prompts_all = list()
    outputs_all = list()
    rewards_all = list()
    for i in tqdm(range(len(test_dataset) // args.batch_size)):
        batch = test_dataset[args.batch_size*i:args.batch_size*(i+1)]
        prompts = [item["prompt"] for item in batch]
        prompts_token = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=args.seq_length,
            padding_side="left",
            return_tensors="pt",
        ).to("cuda:0")

        outputs = policy_model.generate(
            **prompts_token,
            max_new_tokens=(args.seq_length-args.prompt_length),
            return_dict_in_generate=True
        )

        trimmed_outputs = outputs.sequences[
            :, 
            prompts_token["input_ids"].shape[1]:
        ]

        prompts_all += prompts
        outputs_all += [
            tokenizer.decode(item) for item in trimmed_outputs
        ]
        rewards_all += reward_fn(
            prompts_all[-args.batch_size:],
            outputs_all[-args.batch_size:]
        ).tolist()

    cols = [
        "policy_model",
        "reward_model",
        "prompt",
        "completion",
        "reward"
    ]

    df = pd.DataFrame(
        {
            "policy_model": [args.policy_path] * len(prompts_all),
            "reward_model": [args.reward_path] * len(prompts_all),
            "prompt": prompts_all,
            "completion": outputs_all,
            "reward": rewards_all
        }
    )

    os.makedirs(args.eval_result_dir, exist_ok=True)
    df.to_csv(args.eval_result_dir + args.result_name + f"_{args.eval_dataset_size}prompts.csv", index=None)
    
    print("Finished evaluation")

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
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--train_dataset_size", default=None, type=int)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--policy_use_lora", action="store_true")
    parser.add_argument("--reward_use_lora", action="store_true")

    parser.add_argument("--reward_path", type=str, default="ckpts/reward_models/checkpoint-500")
    
    parser.add_argument("--policy_path", type=str, default="ckpts/rlhf/")
    
    parser.add_argument("--policy_data_path", type=str, default="datasets/multilingual-rl-tuning-64k/")

    parser.add_argument("--eval_result_dir", type=str, default="./eval_results/")
    parser.add_argument("--result_name", type=str, default="test0")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    policy_args = get_args()

    set_seed(policy_args.seed)

    print("Training policy model")
    generate_responses(policy_args)

    
