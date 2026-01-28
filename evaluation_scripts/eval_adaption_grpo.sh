lang=$1
train_dataset_size=$2
reward_lora_adapter_path=ckpts/adaption_reward/reward_model_${lang}_${train_dataset_size}/checkpoint-3000
policy_path=ckpts/rlhf/checkpoint-3000
policy_output_dir=ckpts/adaption_reward/policy_${lang}_${train_dataset_size}
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaption_reward.py \
        --reward_lora_adapter_path=$reward_lora_adapter_path \
        --policy_path=$policy_path \
        --policy_output_dir=$policy_output_dir \
        --policy_data_path=$policy_data_path \
        --train_dataset_size=$train_dataset_size\
        --batch_size 2\
        --max_steps=3000 \
        --eval_freq=500 \
        --save_freq=300