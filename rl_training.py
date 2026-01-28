import os, json
import argparse
from typing import List
from dataclasses import field

import time

from other_peft_model_to_sequence_classification import ConvertedPeftModelForSequenceClassification

import torch

from datasets import load_dataset, Dataset, concatenate_datasets
from dataset_util import create_datasets_metagrpo
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    ApertusForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    BloomForCausalLM,
    AutoModelForCausalLM,
    Gemma3ForSequenceClassification,
    Gemma3TextConfig,
    Gemma3ForCausalLM,
)

from peft import (
    prepare_model_for_kbit_training, 
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model,
    cast_mixed_precision_params,
)
# from optimum.bettertransformer import BetterTransformer

from trl import (
    GRPOConfig, 
    GRPOTrainer,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from utils import Prompter
prompter = Prompter()

def create_reward_fn(args, reward_model = None, reward_tokenizer = None):
    print("Setup reward model")
    # if os.environ.get("RANK", "0") == "0":
    if reward_model is None or reward_tokenizer is None:
        model_type = Gemma3ForSequenceClassification if "gemma" in args.reward_path else AutoModelForSequenceClassification
        if args.reward_use_lora:
            reward_model, reward_tokenizer = load_quantized_reward_model(model_type, args.reward_path)
        else:
            reward_model, reward_tokenizer = load_reward_model(model_type, args.reward_path)
    
    if "Llama" in reward_model.config.architectures[0]:
        print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        reward_tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }
        )
        reward_tokenizer.pad_token_id = (
            0
        )
        
    reward_tokenizer.truncation_side = "left"
    reward_tokenizer.padding_side = "left"

    reward_model.requires_grad_(False)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.config.use_cache = False
    reward_model.config.gradient_checkpointing = False

    reward_model.eval()

    
    # sigmoid_fn = nn.Sigmoid()
    def get_reward(samples: List[str]):
        all_scores = []
        for i in range(0, len(samples), args.batch_size):
            batch = reward_tokenizer(
                samples[i : i + args.batch_size],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt").to(device)
            with torch.no_grad():
                scores = reward_model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )[0].squeeze(-1).cpu()
            all_scores.append(scores)
        scores = torch.hstack(all_scores)
        
        return scores

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> torch.Tensor:
        samples = [p + c for p, c in zip(prompts, completions)]
        
        # rewards = get_reward(prompts)
        rewards = get_reward(samples)
        return rewards

    return reward_fn

def load_reward_model(reward_type, reward_path):
    print(f"### model path: {reward_path}")
    tokenizer = AutoTokenizer.from_pretrained(reward_path)
    
    model = reward_type.from_pretrained(
        reward_path,
        device_map=device,
        low_cpu_mem_usage=True,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def load_model(model_type, model_path):
    print(f"### model path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"
    
    model = model_type.from_pretrained(
        model_path,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def load_quantized_reward_model(reward_type, lora_adapter_path):
    try:
        peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    except:
        peft_config = PeftConfig.from_pretrained(lora_adapter_path, subfolder="trainable")
    peft_config.inference_mode = True
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if 'gemma' in lora_adapter_path or 'apertus' in lora_adapter_path:
        base_type = Gemma3ForCausalLM if 'gemma' in lora_adapter_path else AutoModelForCausalLM
        model = base_type.from_pretrained(
            lora_adapter_path,
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation='eager',
            quantization_config=quantization_config
            )
        model.lm_head = torch.nn.Linear(model.config.hidden_size, 1, bias=False,device=device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = prepare_model_for_kbit_training(model)
        # model = PeftGemma3ForSequenceClassification(peft_config, model)
        try:
            model = PeftModel.from_pretrained(model, lora_adapter_path, config=peft_config, is_trainable=True)
        except:
            model = PeftModel.from_pretrained(model, lora_adapter_path, config=peft_config, is_trainable=True, subfolder="trainable")
        model.__class__ = ConvertedPeftModelForSequenceClassification
        model.num_labels = 1
        # model.problem_type == "single_label_classification"
    else:
        model = reward_type.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map=device,
            low_cpu_mem_usage=True,
            num_labels=1,
            quantization_config=quantization_config,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = prepare_model_for_kbit_training(model)
        # model = model.to_bettertransformer()
        try:
            model = PeftModel.from_pretrained(model, lora_adapter_path, config=peft_config, is_trainable=True)
        except:
            model = PeftModel.from_pretrained(model, lora_adapter_path, config=peft_config, is_trainable=True, subfolder="trainable")
    
    
    print("--------------------Reward model: Trainable parameters -------------------")
    model.print_trainable_parameters()
    print("--------------------END -------------------")
    
    return model, tokenizer

def create_quantized_lora_model(model_type, model_path):
    # Quantization + LoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model_type.from_pretrained(
            model_path,
            # trust_remote_code=True,
            device_map=device,
            low_cpu_mem_usage=True,
            # use_cache=False,
            quantization_config=quantization_config,
        )
    print("Model initialized: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    if 'bloom' in model_path:
        modules =  ["query_key_value"]
    elif "Apertus" in model_path:
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
    return model, tokenizer

def create_quantized_lora_reward_model(reward_type, reward_path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = reward_type.from_pretrained(
        reward_path,
        device_map=device,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )


    return model

def load_quantized_lora_model(model_type, lora_adapter_path):
    try:
        peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    except:
        peft_config = PeftConfig.from_pretrained(lora_adapter_path, subfolder="trainable")
    peft_config.inference_mode = False
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = model_type.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map=device,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    # model = model.to_bettertransformer()
    try:
        model = PeftModel.from_pretrained(model, lora_adapter_path, config=peft_config, is_trainable=True)
    except:
        model = PeftModel.from_pretrained(model, lora_adapter_path, config=peft_config, is_trainable=True, subfolder="trainable")
    
    print("--------------------Policy model: Trainable parameters -------------------")
    model.print_trainable_parameters()
    print("--------------------END -------------------")
    
    return model, tokenizer

class BloomWithLogitsToKeep(BloomForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if logits_to_keep is not None:
            output.logits = output.logits[:, -logits_to_keep:, :]

        return output

# language_list = ['ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'it', 'kn', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'ta', 'te', 'uk', 'vi', 'zh']
# language_list = ['ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'it', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'ta', 'te', 'uk', 'vi', 'zh']
language_list = ['ro', 'it', 'es', 'ca']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_lora_adapter_path", type=str, default="ckpts/sft_models/")
    parser.add_argument("--data_path", type=str, default="datasets/multilingual-rl-tuning-64k/")
    parser.add_argument("--reward_path", type=str, default="ckpts/reward_models/")
    parser.add_argument("--language_list", type=str, nargs="+", default=language_list, help='Split by space')

    # parser.add_argument("--rw_batch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=float, default=0.1)
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=512)

    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_layers_unfrozen", type=int, default=5)
    parser.add_argument("--ppo_batch_size", type=int, default=2)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--kl_coef", type=int, default=0.1)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--wandb_name", default='RL-training', type=str)

    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--eval_freq", default=300, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--eval_dataset_size", default=100, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--output_dir", type=str, default="./ckpts/rlhf/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward_use_lora", action="store_true")
    parser.add_argument("--policy_use_lora", action="store_true")

    # parser.add_argument("--hub_model_id", default='rlhf.debug', type=str)

    args = parser.parse_args()
    set_seed(args.seed)
    
    reward_fn = create_reward_fn(args)

    # train_dataset, eval_dataset = create_datasets(args)
    train_dataset, valid_dataset, test_dataset = create_datasets_metagrpo(args)
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_epochs,

        max_steps = args.max_steps,
        eval_strategy="steps",
        eval_steps = args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_generations=args.num_rollouts,

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        # warmup_steps=args.num_warmup_steps,
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
    
    if "gemma" in args.sft_lora_adapter_path:
        model_type = Gemma3ForCausalLM 
    elif "bloom" in args.sft_lora_adapter_path:
        model_type = BloomWithLogitsToKeep
    else:
        model_type = AutoModelForCausalLM
        
    if args.policy_use_lora:
        policy_model, tokenizer = load_quantized_lora_model(model_type, args.sft_lora_adapter_path)
    else:
        policy_model, tokenizer = load_model(model_type, args.sft_lora_adapter_path)

    trainer = GRPOTrainer(
        model=policy_model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print('Finish training')
    print(f"GRPO training time: {(end_time - start_time) / 60:.2f} minutes")

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(args.output_dir, state_dict=cpu_state_dict)  # noqa

    print("Model saved")
    # push to huggingface
    # model = trainer.accelerator.unwrap_model(trainer.model)
    # if accelerator.is_main_process:
    #     model.save_pretrained(args.output_dir)
