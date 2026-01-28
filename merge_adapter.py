import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_save(base_model: str, adapter_path: str, output_path: str):
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model)

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    print("Done! Merged model is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="bigscience/bloom-1b7")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter_path, args.output_path)
