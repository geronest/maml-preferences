#!bin/bash

lang=ro

policy_path=google/gemma-3-270m

#------------MAML-DPO------------

data_path=datasets/multilingual-ranking-data-42k/
LEARNING_RATE=1e-5
INNER_LR_COEF=3
NUM_STEPS=300
EVAL_FREQ=30
SAVE_FREQ=10
BATCH_SIZE=2
EVAl_BATCH_SIZE=2
ACCUMULATION=40
NUM_TASKS_PER_BATCH=2
WANDB_NAME=meta_dpo_gemma_${lang}_lr${LEARNING_RATE}_inner${INNER_LR_COEF}_${NUM_STEPS}steps_innerbatch${BATCH_SIZE}_outerbatch${NUM_TASKS_PER_BATCH}_acc${ACCUMULATION}
policy_dir=ckpts/meta_dpo_gemma/${WANDB_NAME}

# MODIFY language_list if target language changes
CUDA_VISIBLE_DEVICES=0 python dpo.py \
    --use_same_languages_for_eval \
    --language_list ca fr it es \
    --model_path=$policy_path \
    --inner_train_batch_size=${BATCH_SIZE} \
    --eval_batch_size=${EVAl_BATCH_SIZE} \
    --data_path=$data_path \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=1000 \
    --output_dir=${policy_dir} \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --eval_freq=${EVAL_FREQ} \
    --inner_lr_coef=${INNER_LR_COEF} \
    --num_tasks_per_batch=${NUM_TASKS_PER_BATCH} \



#------------Adaptation-DPO------------
SFT_SEARCH_CKPT=$(find ${policy_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
policy_path=${SFT_SEARCH_CKPT}

data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=1e-4
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=2
BETA=0.1
GRAD_ACC_STEP=2
WANDB_NAME=adapt_dpo_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC_STEP}
policy_output_dir=ckpts/meta_dpo_gemma/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python adaptation_dpo.py \
    --model_path=$policy_path \
    --batch_size=${BATCH_SIZE} \
    --data_path=$data_path \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=1000 \
    --eval_freq=${EVAL_FREQ} \
    --output_dir=$policy_output_dir \
    --beta=${BETA} \
    --gradient_accumulation_steps=${GRAD_ACC_STEP} \


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