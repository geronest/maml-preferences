#!bin/bash

export WANDB_API_KEY=b31f5ed0767abb05596b308c3c5efc4384086962
export WANDB_PROJECT=maml-rlhf

lang=ca
# Common Parameters
model_path=bigscience/bloom-7b1
reward_model_path=bigscience/bloom-7b1

GPU_ALLOC=1

# SFT
LEARNING_RATE=1e-4
NUM_STEPS=100   
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=40
TRAINING_DATA_SIZE=8000
GRAD_ACC_STEPS=1
WANDB_NAME=${lang}_bloom7b1_baselinesft_lora_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC_STEPS}
SFT_NUM_STEPS=${NUM_STEPS}

sft_output_dir=ckpts/sft_models/${WANDB_NAME}
merged_sft_output_dir=ckpts/sft_models/merged_${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k/${lang}.json

LANGUAGE_LIST="${lang}"
CUDA_VISIBLE_DEVICES=$GPU_ALLOC python supervised_finetuning.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --train_dataset_size=${TRAINING_DATA_SIZE} \
        --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
        --wandb_name=${WANDB_NAME} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --use_lora

adapter_path=$(find ${sft_output_dir} -type d | sort | tail -n 1)
CUDA_VISIBLE_DEVICES=$GPU_ALLOC python merge_adapter.py \
  --base_model=${model_path} \
  --adapter_path=${adapter_path} \
  --output_path=${merged_sft_output_dir}

# Reward Modeling
reward_path_load=$merged_sft_output_dir
data_path=datasets/multilingual-ranking-data-42k/
reward_data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=1e-5
NUM_STEPS=200
EVAL_FREQ=20
SAVE_FREQ=200
TRAIN_DATA_SIZE=8000
BATCH_SIZE=40
GRAD_ACC=1
WEIGHT_DECAY=1e-6
LR_SCHEDULER_TYPE=constant
WANDB_NAME=${lang}_bloom7b1_baselinerm_sft${SFT_NUM_STEPS}_decay${WEIGHT_DECAY}_lora_lr${LR_SCHEDULER_TYPE}${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC}_tdata${TRAIN_DATA_SIZE}
rm_output_dir=ckpts/reward_models/${WANDB_NAME}
RM_NUM_STEPS=${NUM_STEPS}
RM_BATCH_SIZE=${BATCH_SIZE}
RM_LEARNING_RATE=${LEARNING_RATE}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python adaptation_reward.py \
    --reward_path=$reward_path_load \
    --reward_output_dir=$rm_output_dir \
    --reward_data_path=$reward_data_path \
    --train_dataset_size=${TRAIN_DATA_SIZE} \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --eval_dataset_size=1000 \
    --save_freq=${SAVE_FREQ} \
    --wandb_name=${WANDB_NAME} \
    --gradient_accumulation_steps=${GRAD_ACC} \
    --weight_decay=${WEIGHT_DECAY} \
    --batch_size=${BATCH_SIZE} \
    --eval_steps=100 \
    --lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    --use_lora

# GRPO

REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"

reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

LEARNING_RATE=1e-4
TRAIN_DATA_SIZE=8000
LR_SCHEDULER_TYPE=constant
WEIGHT_DECAY=0.05
NUM_STEPS=100
EVAL_FREQ=100
SAVE_FREQ=100
PROMPT_LEN=128
SEQ_LEN=512
BATCH_SIZE=4
NUM_GENERATIONS=4
GRAD_ACC_STEPS=4
WANDB_NAME=${lang}_bloom7b1_lora_sft${SFT_NUM_STEPS}lr1e-4_sftrm${RM_NUM_STEPS}rlr${RM_LEARNING_RATE}batch${RM_BATCH_SIZE}tdata${TRAIN_DATA_SIZE}_grpo_lr${LR_SCHEDULER_TYPE}${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_decay${WEIGHT_DECAY}

reward_path=${REWARD_SEARCH_CKPT}
sft_path=${SFT_SEARCH_CKPT}

policy_output_dir=ckpts/adaption_grpo/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python adaptation_grpo.py \
        --reward_path=$reward_path \
        --policy_path=$sft_path \
        --policy_output_dir=$policy_output_dir \
        --policy_data_path=$policy_data_path \
        --train_dataset_size=${TRAIN_DATA_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --weight_decay=${WEIGHT_DECAY} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --prompt_length=${PROMPT_LEN} \
        --wandb_name=${WANDB_NAME} \
        --seq_length=${SEQ_LEN} \
        --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
        --bf16 \
        --policy_use_lora \
        --reward_use_lora \
        --lr_scheduler_type=${LR_SCHEDULER_TYPE}

policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

BATCH_SIZE=16
PROMPT_LEN=128
SEQ_LEN=512
EVAL_SIZE=1000

reward_path=ckpts/reward_models/bloom7b1_judgerm_decay1e-6_${lang}_lr5e-5_10000steps_batch16_acc1/checkpoint-10000
policy_path=${policy_output_dir}/checkpoint-${NUM_STEPS}
RESULT_NAME=${lang}/${lang}_${WANDB_NAME}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python eval_with_reward.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --policy_data_path=$policy_data_path \
        --batch_size=${BATCH_SIZE} \
        --prompt_length=${PROMPT_LEN} \
        --seq_length=${SEQ_LEN} \
        --eval_dataset_size=${EVAL_SIZE} \
        --result_name=${RESULT_NAME} \
        --bf16

python3 summarize_eval.py --lang ${lang}