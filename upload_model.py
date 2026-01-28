import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
# from reward_modeling import load_model as load_rm
from rl_training import load_quantized_lora_model, BloomWithLogitsToKeep, load_reward_model, load_model, load_quantized_reward_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Model to load and upload")
    parser.add_argument("--model_type", required=True, default="policy") # "policy" or "reward"
    parser.add_argument("--upload_path", required=True, help="Huggingface path to upload the model to") 
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    print(f"Loading and uploading ### {args.model_type} ### model")

    if args.model_type == "policy":
        if args.use_lora:
            model, tokenizer = load_quantized_lora_model(
                BloomWithLogitsToKeep, 
                args.checkpoint_path
            )
        else:
            model, tokenizer = load_model(
                BloomWithLogitsToKeep, 
                args.checkpoint_path
            )
    elif args.model_type == "reward":
        if args.use_lora:
            model, tokenizer = load_quantized_reward_model(
                AutoModelForSequenceClassification, 
                args.checkpoint_path
            )
        else:
            model, tokenizer = load_reward_model(
                AutoModelForSequenceClassification, 
                args.checkpoint_path
            )

    print("Trying to upload: ", args.upload_path)
    model.push_to_hub(args.upload_path)
    print("#### Upload complete ####")