#!bin/bash

export WANDB_API_KEY=4463d49cb6d35aa2dbab0fe629f9aaeefc8d8898
export WANDB_PROJECT=maml-rlhf-gemma

# Common Parameters
model_path=google/gemma-3-270m

# SFT
LEARNING_RATE=5e-7
INNER_LEARNING_RATE=5e-5
NUM_STEPS=3000
# NUM_EPOCHS=3
EVAL_FREQ=300
SAVE_FREQ=600
# NUM_STEPS=50
# # NUM_EPOCHS=3
# EVAL_FREQ=20
BATCH_SIZE=2
WANDB_NAME=sft_gemma_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}

# sft_output_dir=ckpts/sft_models/${WANDB_NAME}
sft_output_dir=ckpts/sft_models/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
        meta_supervised_finetuning.py \
        --use_same_languages_for_eval \
        --language_list ro ca it es \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --inner_train_batch_size=2 \
        --micro_batch_size=${BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --inner_lr=${INNER_LEARNING_RATE} \
        --wandb_name=${WANDB_NAME} \
        --num_tasks_per_batch 4 \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_steps=100

# Reward Modeling
data_path=datasets/multilingual-ranking-data-42k/
LEARNING_RATE=5e-7
INNER_LEARNING_RATE=5e-5
NUM_STEPS=3
EVAL_FREQ=300
SAVE_FREQ=600
NUM_TASKS_PER_BATCH=2
BATCH_SIZE=2
OUTER_BATCH_SIZE=2
WANDB_NAME=rm_gemma_${NUM_STEPS}steps_lr${LEARNING_RATE}
rm_output_dir=ckpts/reward_models/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --use_same_languages_for_eval \
    --language_list ro ca it es \
    --num_tasks_per_batch ${NUM_TASKS_PER_BATCH} \
    --inner_train_batch_size ${BATCH_SIZE} \
    --micro_batch_size=${OUTER_BATCH_SIZE} \
    --reward_path=$model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --inner_lr=${INNER_LEARNING_RATE} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --eval_steps=100 \

# # GRPO
REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"

# lang=fr
reward_data_path=datasets/multilingual-ranking-data-42k/
policy_data_path=datasets/multilingual-rl-tuning-64k/

LEARNING_RATE=5e-6
WEIGHT_DECAY=0.05
NUM_STEPS=3000
EVAL_FREQ=300
SAVE_FREQ=600
# NUM_STEPS=50
# EVAL_FREQ=20
PROMPT_LEN=128
SEQ_LEN=512
BATCH_SIZE=2
NUM_GENERATIONS=4
GRAD_ACC_STEPS=4
WANDB_NAME=rl_gemma_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_decay${WEIGHT_DECAY}

reward_path=${REWARD_SEARCH_CKPT}
sft_path=${SFT_SEARCH_CKPT}
# sft_path=google/gemma-3-270m
# reward_path=google/gemma-3-270m

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
    --max_new_tokens=${SEQ_LEN} \
    --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
    # --reward_use_lora
