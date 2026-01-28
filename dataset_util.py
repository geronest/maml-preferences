import os, json
import random
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from utils import Prompter
prompter = Prompter()

def tokenize_metasft(args, tokenizer, data_point, add_eos_token=True):
    prompt = [prompter.generate_prompt(instr, input, output)
              for instr, input, output in
              zip(data_point["instruction"], data_point["input"], data_point["output"])]

    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.seq_length,
        padding="max_length",
        padding_side="right",
        return_tensors=None,
    )

    for i in range(len(result["input_ids"])):
        if (
            result["input_ids"][i][-1] != tokenizer.eos_token_id
            and len(result["input_ids"][i]) < args.seq_length
            and add_eos_token
        ):
            result["input_ids"][i].append(tokenizer.eos_token_id)
            result["attention_mask"][i].append(1)

    result["labels"] = [ids.copy() for ids in result["input_ids"]]

    return result
    
def tokenize_sft(args, tokenizer, prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.seq_length,
        padding="max_length",
        padding_side="right",
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < args.seq_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def tokenize_reward(args, tokenizer, data_point, add_eos_token=True):
    chosen = [prompter.generate_prompt(instr, input, preferred)
              for instr, input, preferred in
              zip(data_point["instruction"], data_point["input"], data_point["prefered_output"])]

    rejected = [prompter.generate_prompt(instr, input, rejected)
              for instr, input, rejected in
              zip(data_point["instruction"], data_point["input"], data_point["rejected_output"])]

    tokenized_chosen = tokenizer(
        chosen,
        padding='max_length',
        truncation=True,
        max_length=args.seq_length,
        return_tensors="pt",
    )
    tokenized_rejected = tokenizer(
        rejected,
        padding='max_length',
        truncation=True,
        max_length=args.seq_length,
        return_tensors="pt",
    )

    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

def tokenize_dpo(args, tokenizer, data_point):
    prompts = [prompter.generate_prompt(instr, input) 
               for instr, input in 
               zip(data_point["instruction"], data_point["input"])]

    chosen_full = [prompter.generate_prompt(instr, input, preferred)
                   for instr, input, preferred in
                   zip(data_point["instruction"], data_point["input"], data_point["prefered_output"])]

    rejected_full = [prompter.generate_prompt(instr, input, rejected)
                     for instr, input, rejected in
                     zip(data_point["instruction"], data_point["input"], data_point["rejected_output"])]

    
    tokenized_prompts = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=args.seq_length,
        return_tensors="pt",
    )
    
    tokenized_chosen = tokenizer(
        chosen_full,
        padding='max_length',
        truncation=True,
        max_length=args.seq_length,
        return_tensors="pt",
    )
    
    tokenized_rejected = tokenizer(
        rejected_full,
        padding='max_length',
        truncation=True,
        max_length=args.seq_length,
        return_tensors="pt",
    )

    return {
        "prompt_input_ids": tokenized_prompts["input_ids"],
        "prompt_attention_mask": tokenized_prompts["attention_mask"],
        
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
    }

def generate_and_tokenize_prompt(args, tokenizer, dataset, language_list, func_tokenize):
    # Tokenize all data for each task
    ds = {}
    for language in language_list:
        ds[language] = dataset[language].map(
            lambda x: func_tokenize(args, tokenizer, x),
            batched=True,
            remove_columns=dataset[language].column_names,
            num_proc=16
        )

    return DatasetDict(ds)

def generate_dataset(args, languages, type_model="sft"):
    # Generate a whole dataset that contains all tasks, 
    # each task has a complete dataset.
    dataset = {}

    if type_model == "sft":
        cols_select = ['instruction','input', 'output']
    elif type_model in ["reward", "dpo"] :
        cols_select = ['instruction','input', 'prefered_output', 'rejected_output']
    
    for language in languages:
        full_path = os.path.join(args.data_path, f'{language}.json')
        raw_ds = load_dataset('json', split=args.split, data_files=full_path)
        ds = raw_ds.select_columns(cols_select)
        dataset[language] = ds
    return DatasetDict(dataset)

def generate_train_test_tasks(args):
    # Train dataset: contains 90% (flexible) tasks; each task still has a complete dataset
    # Eval dataset: contains 10% (flexible) tasks; each task still has a complete dataset
    language_list = args.language_list
    
    if args.use_same_languages_for_eval:
        train_languages = language_list
        eval_languages = language_list
        datasets = generate_dataset(args, train_languages, type_model=args.type_model)
        for k in datasets:
            datasets[k] = datasets[k].train_test_split(
                test_size=args.size_valid_set, seed=args.seed
            )
        train_dataset = {
            k: datasets[k]["train"] for k in datasets
        }
        eval_dataset = {
            k: datasets[k]["test"] for k in datasets
        }
        for k in datasets:
            eval_dataset[k] = eval_dataset[k].train_test_split(
                test_size=0.5, seed=args.seed
            )
        valid_dataset = {
            k: eval_dataset[k]["train"] for k in datasets
        }
        test_dataset = {
            k: eval_dataset[k]["test"] for k in datasets
        }
    else:
        random.seed(args.seed)
        k = int(args.size_valid_set * len(language_list))
        eval_languages = random.sample(language_list, k)
        train_languages = [lang for lang in language_list if lang not in eval_languages]
        train_dataset = generate_dataset(args, train_languages)
        eval_dataset = generate_dataset(args, eval_languages)
    train_dataset = {
        k: train_dataset[k].shuffle() for k in train_dataset
    }
    eval_dataset = {
        k: eval_dataset[k].shuffle() for k in eval_dataset
    }
    valid_dataset = {
        k: valid_dataset[k].shuffle() for k in valid_dataset
    }
    test_dataset = {
        k: test_dataset[k].shuffle() for k in test_dataset
    }
    return train_languages, eval_languages, train_dataset, valid_dataset, test_dataset
    # return train_languages, eval_languages, train_dataset, eval_dataset

# Meta - SFT
def create_datasets_metasft(tokenizer, args):
    """Create the datasets for training and validation."""
    
    train_languages, eval_languages, train_dataset, valid_dataset, test_dataset = generate_train_test_tasks(args)
    prompter = Prompter()
    train_dataset = generate_and_tokenize_prompt(args, tokenizer, train_dataset, train_languages, tokenize_metasft)
    valid_dataset = generate_and_tokenize_prompt(args, tokenizer, valid_dataset, eval_languages, tokenize_metasft)
    test_dataset = generate_and_tokenize_prompt(args, tokenizer, test_dataset, eval_languages, tokenize_metasft)
    
    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, valid_dataset, test_dataset

# Meta - Reward
def create_datasets_metareward(args, tokenizer):
    """Create the datasets for training and validation."""

    train_languages, eval_languages, train_dataset, valid_dataset, test_dataset = generate_train_test_tasks(args)
    prompter = Prompter()
    train_dataset = generate_and_tokenize_prompt(args, tokenizer, train_dataset, train_languages, tokenize_reward)
    valid_dataset = generate_and_tokenize_prompt(args, tokenizer, valid_dataset, eval_languages, tokenize_reward)
    test_dataset = generate_and_tokenize_prompt(args, tokenizer, test_dataset, eval_languages, tokenize_reward)

    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, valid_dataset, test_dataset

# Meta - DPO
def create_datasets_metadpo(args, tokenizer):
    """Create the datasets for training and validation."""

    train_languages, eval_languages, train_dataset, valid_dataset, test_dataset = generate_train_test_tasks(args)
    prompter = Prompter()
    train_dataset = generate_and_tokenize_prompt(args, tokenizer, train_dataset, train_languages, tokenize_dpo)
    valid_dataset = generate_and_tokenize_prompt(args, tokenizer, valid_dataset, eval_languages, tokenize_dpo)
    test_dataset = generate_and_tokenize_prompt(args, tokenizer, test_dataset, eval_languages, tokenize_dpo)

    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, valid_dataset, test_dataset

# Meta - GRPO
def create_datasets_metagrpo(args):
    def create_prompt(data_point):
        prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            # data_point["output"],
        )
        return {'prompt': prompt}

    prompter = Prompter()
    all_datasets = []
    for lang in args.language_list:
        full_path = os.path.join(args.data_path, f'{lang}.json')
        try:
            dataset = load_dataset('json', split=args.split, data_files=full_path)
        except:
            with open(full_path, 'r', encoding='utf-8') as f:
                dataset = json.loads(f.read())
            for entry in dataset:
                for k, v in entry.items():
                    if not isinstance(v, str):
                        entry[k] = str(v)
            dataset = Dataset.from_list(dataset)

        dataset = dataset.select_columns(['instruction','input', 'output'])
        all_datasets.append(dataset)

    full_dataset = concatenate_datasets(all_datasets)

    dataset = full_dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)
    dataset["test"] = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)

    train_prompts = dataset["train"].shuffle(seed=args.seed).map(create_prompt)
    valid_prompts = dataset["test"]["train"].shuffle(seed=args.seed).map(create_prompt).select(list(range(args.eval_dataset_size)))
    test_prompts = dataset["test"]["test"].shuffle(seed=args.seed).map(create_prompt)

    train_prompts = [{'prompt': instance['prompt']} for instance in train_prompts]
    valid_prompts = [{'prompt': instance['prompt']} for instance in valid_prompts]
    test_prompts = [{'prompt': instance['prompt']} for instance in test_prompts]

    print(f"Size of the train set: {len(train_prompts)}. Size of the validation set: {len(valid_prompts)}")

    return train_prompts, valid_prompts, test_prompts

# Baseline - SFT
def create_datasets_sft(tokenizer, args):
    """Create the datasets for training and validation."""

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize_sft(args, tokenizer, full_prompt)
        return tokenized_full_prompt

    prompter = Prompter()
    dataset = load_dataset('json', split=args.split, data_files=args.data_path)
    original_columns = dataset.column_names
    

    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)
    dataset["test"] = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)

    train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=128, remove_columns=original_columns).select(list(range(args.train_dataset_size)))
    valid_data = dataset["test"]["train"].map(generate_and_tokenize_prompt, remove_columns=original_columns).select(list(range(args.eval_dataset_size)))
    test_data = dataset["test"]["test"].map(generate_and_tokenize_prompt, remove_columns=original_columns)
    train_data.set_format("torch")
    valid_data.set_format("torch")
    test_data.set_format("torch")

    # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    return train_data, valid_data, test_data

# Baseline - Reward
def load_reward_dataset(args, tokenizer):

    dataset = load_dataset('json', split='train', data_files=args.reward_data_path)
    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)
    dataset["test"] = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)

    prompter = Prompter()
    
    if args.train_dataset_size is not None:
        train_dataset = dataset['train'].shuffle().map(
            # tokenize_reward,
            lambda x: tokenize_reward(args, tokenizer, x),
            batched=True,
            remove_columns=dataset['train'].column_names,
            num_proc=16).select(list(range(args.train_dataset_size)))
    else:
        train_dataset = dataset['train'].shuffle().map(
            # tokenize_reward,
            lambda x: tokenize_reward(args, tokenizer, x),
            batched=True,
            remove_columns=dataset['train'].column_names,
            num_proc=16)
        
    valid_dataset = dataset['test']["train"].shuffle().map(
            # tokenize_reward,
            lambda x: tokenize_reward(args, tokenizer, x),
            batched=True,
            remove_columns=dataset['test']["train"].column_names,
            num_proc=16).select(list(range(args.eval_dataset_size)))
    test_dataset = dataset['test']["test"].shuffle().map(
            # tokenize_reward,
            lambda x: tokenize_reward(args, tokenizer, x),
            batched=True,
            remove_columns=dataset['test']["test"].column_names,
            num_proc=16)

    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, valid_dataset, test_dataset

# Baseline - GRPO
def create_datasets_grpo(args):
    print("Start create_datasets")
    def create_prompt(data_point):
        prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            # data_point["output"],
        )
        return {'prompt': prompt}

    prompter = Prompter()
    try:
        dataset = load_dataset('json', split=args.split, data_files=args.policy_data_path)
    except:
        with open(args.policy_data_path, 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())
        for entry in dataset:
            for k, v in entry.items():
                if not isinstance(v, str):
                    entry[k] = str(v)
        dataset = Dataset.from_list(dataset)

    dataset = dataset.select_columns(['instruction','input', 'output'])

    dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)
    dataset["test"] = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)

    if args.train_dataset_size is not None:
        train_prompts = dataset["train"].shuffle(seed=args.seed).map(create_prompt).select(list(range(args.train_dataset_size)))
    else:
        train_prompts = dataset["train"].shuffle(seed=args.seed).map(create_prompt)
        
    valid_prompts = dataset["test"]["train"].shuffle(seed=args.seed).map(create_prompt).select(list(range(args.eval_dataset_size)))
    test_prompts = dataset["test"]["test"].shuffle(seed=args.seed).map(create_prompt)

    train_prompts = [{'prompt': instance['prompt']} for instance in train_prompts]
    valid_prompts = [{'prompt': instance['prompt']} for instance in valid_prompts]
    test_prompts = [{'prompt': instance['prompt']} for instance in test_prompts]

    print(f"Size of the train set: {len(train_prompts)}. Size of the validation set: {len(valid_prompts)}. Size of the test set: {len(test_prompts)}.")

    return train_prompts, valid_prompts, test_prompts
