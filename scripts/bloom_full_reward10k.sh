#!bin/bash

export WANDB_API_KEY=b31f5ed0767abb05596b308c3c5efc4384086962
export WANDB_PROJECT=maml-rlhf

# used target languages in the paper: ca fr ro
lang=ca
# Common Parameters
model_path=bigscience/bloom-7b1
reward_model_path=bigscience/bloom-7b1

GPU_ALLOC=6

# Reward Modeling
data_path=datasets/multilingual-ranking-data-42k/
reward_data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=5e-5
NUM_STEPS=10000
EVAL_FREQ=1000
SAVE_FREQ=10000
BATCH_SIZE=16
GRAD_ACC=1
WEIGHT_DECAY=1e-6
WANDB_NAME=bloom7b1_judgerm_decay${WEIGHT_DECAY}_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC}
rm_output_dir=ckpts/reward_models/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python adaptation_reward.py \
    --reward_path=$reward_model_path \
    --reward_output_dir=$rm_output_dir \
    --reward_data_path=$reward_data_path \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --wandb_name=${WANDB_NAME} \
    --gradient_accumulation_steps=${GRAD_ACC} \
    --weight_decay=${WEIGHT_DECAY} \
    --batch_size=${BATCH_SIZE} \
    --eval_steps=100
