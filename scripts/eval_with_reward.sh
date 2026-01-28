#!bin/bash

lang=fr
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

BATCH_SIZE=16
PROMPT_LEN=128
SEQ_LEN=512
EVAL_SIZE=1000

lang=fr
reward_path=ckpts/reward_models/bloom7b1_judgerm_decay1e-6_${lang}_lr5e-5_10000steps_batch16_acc1/checkpoint-10000
policy_path=bigscience/bloom-7b1

RESULT_NAME=${lang}_judgerm_bloom7b1-base

GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python eval_with_reward.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --policy_data_path=$policy_data_path \
        --batch_size=${BATCH_SIZE} \
        --prompt_length=${PROMPT_LEN} \
        --seq_length=${SEQ_LEN} \
        --eval_dataset_size=${EVAL_SIZE} \
        --result_name=${RESULT_NAME} \
        --bf16
