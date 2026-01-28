#!bin/bash

export WANDB_API_KEY=4463d49cb6d35aa2dbab0fe629f9aaeefc8d8898
export WANDB_PROJECT=baseline-pipe2-rlhf-bloom_sft

# Common Parameters
model_path=bigscience/bloom-1b7
lang=fr

LEARNING_RATE=$1
NUM_STEPS=3000
# NUM_EPOCHS=3
EVAL_FREQ=150
SAVE_FREQ=600
# NUM_STEPS=50
# # NUM_EPOCHS=3
# EVAL_FREQ=20
BATCH_SIZE=$2
WEIGHT_DECAY=1e-6
WARMUP=$3
WANDB_NAME=sft_pipe2_bloom_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_warmup${WARMUP}
# WANDB_NAME=test_gemma
# sft_output_dir=ckpts/sft_models/${WANDB_NAME}
sft_output_dir=ckpts/sft_models/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k/${lang}.json

CUDA_VISIBLE_DEVICES="0" python sft.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=${WARMUP} \
        --learning_rate=${LEARNING_RATE} \
        --wandb_name=${WANDB_NAME} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_dataset_size=100 \
        --weight_decay=${WEIGHT_DECAY} \
        --use_lora \