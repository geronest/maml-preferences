lang=$1
reward_path=ckpts/reward_models/checkpoint-500
policy_path=ckpts/rlhf/
reward_output_dir=ckpts/adaption_reward/reward_model_$lang
policy_output_dir=ckpts/adaption_reward/policy_$lang
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --reward_output_dir=$reward_output_dir \
        --policy_output_dir=$policy_output_dir \
        --reward_data_path=$reward_data_path \
        --policy_data_path=$policy_data_path \
