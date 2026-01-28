#!bin/bash

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=baseline-rlhf-bloom_rm

# Common Parameters
model_path=bigscience/bloom-1b7
lang=fr

data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=3e-5
NUM_STEPS=3000
EVAL_FREQ=300
SAVE_FREQ=600
BATCH_SIZE=8
WANDB_NAME=rm_bloom_${lang}_${NUM_STEPS}steps_lr${LEARNING_RATE}_batch${BATCH_SIZE}
rm_output_dir=ckpts/reward_models/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --batch_size=${BATCH_SIZE} \
    --model_path=$model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=500 \
    --gradient_accumulation_steps=1 \
    --save_limit=2 \
    --use_lora \
    --weight_decay=1e-6 \
