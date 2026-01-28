#!bin/bash

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=baseline-dpo-gemma

# Common Parameters
model_path=google/gemma-3-270m
lang=ro

# ----------SFT for Gemma-----------
LEARNING_RATE=5e-6
NUM_STEPS=3000
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=4
WEIGHT_DECAY=1e-6
WARMUP=5
WANDB_NAME=sft_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
sft_output_dir=ckpts/gemma_baseline/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k/${lang}.json

CUDA_VISIBLE_DEVICES="0" python supervised_finetuning.py \
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
        --eval_dataset_size=1000 \
        --weight_decay=${WEIGHT_DECAY} \


# ------------DPO-------------
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
sft_path=${SFT_SEARCH_CKPT}

policy_data_path=datasets/multilingual-ranking-data-42k/${lang}.json

LEARNING_RATE=5e-5
WEIGHT_DECAY=0.05
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=2
EVAL_BATCH_SIZE=2
GRAD_ACC_STEPS=2
WANDB_NAME=dpo_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC_STEPS}

policy_output_dir=ckpts/gemma_baseline/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python adaptation_dpo.py \
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

#--------------Evaluation---------------
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

BATCH_SIZE=16
PROMPT_LEN=128
SEQ_LEN=512
EVAL_SIZE=1000

reward_path=Meta-Okapi/${lang}_bloom1b7_judgerm_decay1e-6_lr5e-5_10ksteps
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
