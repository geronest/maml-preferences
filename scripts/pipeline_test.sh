#!bin/bash

export WANDB_API_KEY=b31f5ed0767abb05596b308c3c5efc4384086962
export WANDB_PROJECT=maml-rlhf

# Common Parameters
model_path=bigscience/bloom-1b7

# SFT
LEARNING_RATE=1e-5
NUM_STEPS=1000
# NUM_EPOCHS=3
EVAL_FREQ=300
# NUM_STEPS=200
# # NUM_EPOCHS=3
# EVAL_FREQ=100
BATCH_SIZE=128
WANDB_NAME=testsft_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}

sft_output_dir=ckpts/sft_models/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

LANGUAGE_LIST="fr"
# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
#         supervised_finetuning.py \
#         --model_path=$model_path \
#         --data_path=$data_path \
#         --output_dir=$sft_output_dir \
#         --batch_size=${BATCH_SIZE} \
#         --num_warmup_steps=100 \
#         --learning_rate=${LEARNING_RATE} \
#         --inner_lr=${LEARNING_RATE} \
#         --num_epochs=${NUM_EPOCHS} \
        # --wandb_name=${WANDB_NAME} \
#         --eval_freq=${EVAL_FREQ} \
#         --eval_steps=100 \
#         --num_tasks_per_batch 1 \
#         --language_list ${LANGUAGE_LIST}


LEARNING_RATE=1e-6
NUM_STEPS=10
EVAL_FREQ=5
WANDB_NAME=testsft_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
sft_output_dir=ckpts/testsft_models/${WANDB_NAME}

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
        supervised_finetuning.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=100 \
        --learning_rate=${LEARNING_RATE} \
        --wandb_name=${WANDB_NAME} \
        --inner_lr=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --eval_steps=10 \
        --num_tasks_per_batch 1 \
        --language_list ${LANGUAGE_LIST}


# Reward Modeling
data_path=datasets/multilingual-ranking-data-42k/
reward_data_path=datasets/multilingual-ranking-data-42k/fr.json

LEARNING_RATE=1e-5
# NUM_STEPS=200
# EVAL_FREQ=100
NUM_STEPS=1000
EVAL_FREQ=50
SAVE_FREQ=500
BATCH_SIZE=4
GRAD_ACC=1
WEIGHT_DECAY=1e-6
WANDB_NAME=rm_decay${WEIGHT_DECAY}_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC}
rm_output_dir=ckpts/reward_models/${WANDB_NAME}

# CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
#     --reward_path=$model_path \
#     --reward_output_dir=$rm_output_dir \
#     --reward_data_path=$reward_data_path \
#     --learning_rate=${LEARNING_RATE} \
#     --max_steps=${NUM_STEPS} \
#     --eval_freq=${EVAL_FREQ} \
#     --save_freq=${SAVE_FREQ} \
#     --wandb_name=${WANDB_NAME} \
#     --gradient_accumulation_steps=${GRAD_ACC} \
#     --weight_decay=${WEIGHT_DECAY} \
#     --batch_size=${BATCH_SIZE} \
#     --eval_steps=100 \
#     --num_warmup_steps=100

# GRPO

# LEARNING_RATE=3e-5
# # NUM_STEPS=200
# # EVAL_FREQ=100
# NUM_STEPS=10000
# EVAL_FREQ=50
# SAVE_FREQ=500
# BATCH_SIZE=4
# GRAD_ACC=4
# WEIGHT_DECAY=1e-6
# WANDB_NAME=rm_decay${WEIGHT_DECAY}_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC}
# rm_output_dir=ckpts/reward_models/${WANDB_NAME}


# PATH_SEARCH_CKPT_1=${HYDRA_RUN_DIR}/${EXPERIMENT_FOLDER}/

REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"

lang=fr
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json


# LEARNING_RATE=5e-6
LEARNING_RATE=1e-5
# LEARNING_RATE=4e-6
WEIGHT_DECAY=0.0
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=200
# NUM_STEPS=200
# EVAL_FREQ=100
PROMPT_LEN=128
SEQ_LEN=512
# SEQ_LEN=256
BATCH_SIZE=4
NUM_GENERATIONS=4
# BATCH_SIZE=8
# NUM_GENERATIONS=8
GRAD_ACC_STEPS=4
WANDB_NAME=grpo2_trainedrm_nolora_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_decay${WEIGHT_DECAY}

# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/rm_debug_nolora_fr_lr1e-6_10000steps_batch8/checkpoint-10000
reward_path=${REWARD_SEARCH_CKPT}

# policy_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_nolora_fr_lr1e-6_10000steps_batch16/checkpoint-10000
sft_path=${SFT_SEARCH_CKPT}

policy_output_dir=ckpts/adaption_grpo/${WANDB_NAME}

# CUDA_VISIBLE_DEVICES=0 python adaptation_grpo.py \
#         --reward_path=$reward_path \
#         --policy_path=$sft_path \
#         --policy_output_dir=$policy_output_dir \
#         --policy_data_path=$policy_data_path \
#         --learning_rate=${LEARNING_RATE} \
#         --max_steps=${NUM_STEPS} \
#         --eval_freq=${EVAL_FREQ} \
#         --save_freq=${SAVE_FREQ} \
#         --weight_decay=${WEIGHT_DECAY} \
#         --batch_size=${BATCH_SIZE} \
#         --num_generations=${NUM_GENERATIONS} \
#         --prompt_length=${PROMPT_LEN} \
#         --wandb_name=${WANDB_NAME} \
#         --seq_length=${SEQ_LEN} \
#         --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
#         --bf16

