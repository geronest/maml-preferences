lang=$1
train_dataset_size=$2
reward_path=ckpts/reward_models/rm/checkpoint-3000
reward_output_dir=ckpts/adaption_reward/reward_model_${lang}_${train_dataset_size}
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaption_reward.py \
        --reward_path=$reward_path \
        --reward_output_dir=$reward_output_dir \
        --reward_data_path=$reward_data_path \
        --train_dataset_size=$train_dataset_size\
        --batch_size 2\
        --max_steps=3000 \
        --eval_freq=500 \
        --save_freq=300