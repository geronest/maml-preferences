lang=$1

# reward_path=ckpts/reward_models/checkpoint-500
# reward_path=ckpts/reward_models/rm-zerotrain/checkpoint-1
reward_path=ckpts/adaption_reward/reward_model_fr_3000/checkpoint-3000
reward_output_dir=ckpts/adaption_reward/reward_model_${lang}_3000
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
        --reward_path=$reward_path \
        --reward_output_dir=$reward_output_dir \
        --reward_data_path=$reward_data_path \
        --max_steps 3000 \
        --eval_freq 100
        

