#!bin/bash

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=baseline-largemeta_rlhf-gemma_rl_fr

# Common Parameters
# model_path=google/gemma-3-270m
# lang=fr

# # SFT
# LEARNING_RATE=5e-5
# NUM_STEPS=3000
# # NUM_EPOCHS=3
# EVAL_FREQ=300
# SAVE_FREQ=600
# # NUM_STEPS=50
# # # NUM_EPOCHS=3
# # EVAL_FREQ=20
# BATCH_SIZE=8
# WANDB_NAME=sft_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}

# # sft_output_dir=ckpts/sft_models/${WANDB_NAME}
# sft_output_dir=ckpts/sft_models/${WANDB_NAME}
# data_path=datasets/multilingual-alpaca-52k/${lang}.json

# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
#         sft.py \
#         --model_path=$model_path \
#         --data_path=$data_path \
#         --output_dir=$sft_output_dir \
#         --batch_size=${BATCH_SIZE} \
#         --num_warmup_steps=5 \
#         --learning_rate=${LEARNING_RATE} \
#         --wandb_name=${WANDB_NAME} \
#         --max_steps=${NUM_STEPS} \
#         --eval_freq=${EVAL_FREQ} \
#         --save_freq=${SAVE_FREQ} \

# Reward Modeling
# data_path=datasets/multilingual-ranking-data-42k/${lang}.json
# LEARNING_RATE=5e-5
# NUM_STEPS=3000
# # EVAL_FREQ=300
# # SAVE_FREQ=600
# BATCH_SIZE=8
# WEIGHT_DECAY=1e-6
# WANDB_NAME=rm_gemma_${lang}_${NUM_STEPS}steps_lr${LEARNING_RATE}_batch${BATCH_SIZE}_decay${WEIGHT_DECAY}
# rm_output_dir=ckpts/reward_models/${WANDB_NAME}

# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --batch_size=${BATCH_SIZE} \
#     --reward_path=$model_path \
#     --data_path=$data_path \
#     --output_dir=$rm_output_dir \
#     --num_warmup_steps=5 \
#     --learning_rate=${LEARNING_RATE} \
#     --wandb_name=${WANDB_NAME} \
#     --max_steps=${NUM_STEPS} \
#     --eval_freq=${EVAL_FREQ} \
#     --save_freq=${SAVE_FREQ} \
# sft_output_dir=ckpts/sft_models/sft_gemma_fr_lr5e-5_3000steps_batch8
sft_output_dir=ckpts/sft_models/sft_gemma_fr_lr5e-6_3000steps_batch4_decay1e-6
# rm_output_dir=ckpts/reward_models/rm_bloom_fr_3000steps_lr3e-5_batch4
CKPT=$4
# rm_output_dir=ckpts/reward_models/rm_bloom_base_ca_3000steps_lr5e-5_batch8
rm_output_dir=ckpts/reward_models/${CKPT}
# GRPO
REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"

lang=fr
reward_data_path=datasets/multilingual-ranking-data-42k/${lang}.json
policy_data_path=datasets/multilingual-rl-tuning-64k/${lang}.json

LEARNING_RATE=$1
WEIGHT_DECAY=0.05
NUM_STEPS=$2
EVAL_FREQ=100
SAVE_FREQ=500
# NUM_STEPS=50
# EVAL_FREQ=20
PROMPT_LEN=128
SEQ_LEN=512
BATCH_SIZE=4
EVAL_BATCH_SIZE=${BATCH_SIZE}
NUM_GENERATIONS=4
GRAD_ACC_STEPS=1
name=$3
WANDB_NAME=rl_baseline_largemeta_gemma_${name}_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_generations${NUM_GENERATIONS}_acc${GRAD_ACC_STEPS}

reward_path=${REWARD_SEARCH_CKPT}
sft_path=${SFT_SEARCH_CKPT}

# reward_path=google/gemma-3-270m
# sft_path=google/gemma-3-270m

policy_output_dir=ckpts/adaption_grpo/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python rl_training.py \
    --reward_path=$reward_path \
    --sft_lora_adapter_path=$sft_path \
    --output_dir=$policy_output_dir \
    --data_path=$policy_data_path \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --weight_decay=${WEIGHT_DECAY} \
    --batch_size=${BATCH_SIZE} \
    --wandb_name=${WANDB_NAME} \
    --num_rollouts=${NUM_GENERATIONS} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --max_new_tokens=${SEQ_LEN} \
    --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
    --eval_dataset_size=100 \
    --reward_use_lora \

