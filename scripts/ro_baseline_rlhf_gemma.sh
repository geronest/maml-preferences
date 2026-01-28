#!bin/bash

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=baseline-rlhf-gemma

# Common Parameters
model_path=google/gemma-3-270m
reward_model_path=bigscience/bloom-1b7
lang=ro

#------------SFT for Gemma------------
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



#------------SFT for Bloom-1b7------------
LEARNING_RATE=5e-6
NUM_STEPS=3000
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=4
WEIGHT_DECAY=1e-6
WARMUP=5
WANDB_NAME=sft_bloom1b7_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
rm_sft_output_dir=ckpts/gemma_baseline/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k/${lang}.json

CUDA_VISIBLE_DEVICES="0" python supervised_finetuning.py \
        --model_path=$reward_model_path \
        --data_path=$data_path \
        --output_dir=$rm_sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=${WARMUP} \
        --learning_rate=${LEARNING_RATE} \
        --wandb_name=${WANDB_NAME} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_dataset_size=1000 \
        --weight_decay=${WEIGHT_DECAY} \
        --use_lora \


adapter_path=$(find ${rm_sft_output_dir} -type d | sort | tail -n 1)

CUDA_VISIBLE_DEVICES=6 python merge_adapter.py \
  --base_model=${reward_model_path} \
  --adapter_path=${adapter_path} \
  --output_path=${rm_sft_output_dir}


#--------Reward Modeling----------
data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=5e-4
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=8
ACCUMULATION=5
WANDB_NAME=rm_bloom1b7_${lang}_${NUM_STEPS}steps_acc${ACCUMULATION}_lr${LEARNING_RATE}_batch${BATCH_SIZE}
rm_output_dir=ckpts/gemma_baseline/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
    --batch_size=${BATCH_SIZE} \
    --model_path=$model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=1000 \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --save_limit=2 \
    --use_lora \
    --weight_decay=1e-6 \

#-------------GRPO----------------
REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
reward_path=${REWARD_SEARCH_CKPT}
sft_path=${SFT_SEARCH_CKPT}

policy_data_path=datasets/multilingual-rl-tuning-64k/${lang}.json

LEARNING_RATE=1e-5
WEIGHT_DECAY=0.05
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=100
# NUM_STEPS=50
# EVAL_FREQ=20
PROMPT_LEN=128
SEQ_LEN=512
BATCH_SIZE=4
EVAL_BATCH_SIZE=${BATCH_SIZE}
NUM_GENERATIONS=4
GRAD_ACC_STEPS=1
WANDB_NAME=baseline_rl_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC_STEPS}_generations${NUM_GENERATIONS}

policy_output_dir=ckpts/gemma_baseline/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python adaptation_grpo.py \
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


#---------evaluation----------
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

BATCH_SIZE=16
PROMPT_LEN=128
SEQ_LEN=512
EVAL_SIZE=1000

reward_path=Meta-Okapi/${lang}_bloom1b7_judgerm_decay1e-6_lr5e-5_10ksteps
policy_path=${policy_output_dir}/checkpoint-${NUM_STEPS}
RESULT_NAME=${lang}/${lang}_${WANDB_NAME}

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

python3 summarize_eval.py --lang ${lang}
