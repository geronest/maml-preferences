#!bin/bash

export WANDB_API_KEY=4463d49cb6d35aa2dbab0fe629f9aaeefc8d8898
export WANDB_PROJECT=baseline-rlhf-gemma_dpo

# sft_output_dir=ckpts/sft_models/sft_gemma_fr_lr5e-6_3000steps_batch4_decay1e-6
sft_output_dir=google/gemma-3-270m
# DPO
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"

lang=fr
policy_data_path=datasets/multilingual-ranking-data-42k/${lang}.json

LEARNING_RATE=$1
WEIGHT_DECAY=0.05
NUM_STEPS=$2
EVAL_FREQ=100
SAVE_FREQ=100
# NUM_STEPS=50
# EVAL_FREQ=20
BATCH_SIZE=2
EVAL_BATCH_SIZE=2
GRAD_ACC_STEPS=$3
WANDB_NAME=dpo_largebatch_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC_STEPS}_base

reward_path=${REWARD_SEARCH_CKPT}
sft_path=${SFT_SEARCH_CKPT}
sft_path=google/gemma-3-270m

# reward_path=google/gemma-3-270m
# sft_path=google/gemma-3-270m

policy_output_dir=ckpts/dpo/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python dpo.py \
    --model_path=${sft_path} \
    --output_dir=${policy_output_dir} \
    --data_path=${policy_data_path} \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --weight_decay=${WEIGHT_DECAY} \
    --batch_size=${BATCH_SIZE} \
    --wandb_name=${WANDB_NAME} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
    --eval_dataset_size=1000 \

