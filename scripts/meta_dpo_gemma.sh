#!bin/bash

target_lang=ro

# MAML-DPO
policy_path=google/gemma-3-270m

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
adapt_rm_dir=ckpts/meta_dpo/maml_dpo_gemma_${target_lang}

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
    --output_dir=${adapt_rm_dir} \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --eval_freq=${EVAL_FREQ} \
    --inner_lr_coef=${INNER_LR_COEF} \
    --num_tasks_per_batch=${NUM_TASKS_PER_BATCH} \



# Adaptation-DPO
policy_dir=ckpts/meta_dpo/maml_dpo_gemma_${target_lang}/
SFT_SEARCH_CKPT=$(find ${policy_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
policy_path=${SFT_SEARCH_CKPT}

data_path=datasets/multilingual-ranking-data-42k/${target_lang}.json
LEARNING_RATE=1e-4
NUM_STEPS=100
EVAL_FREQ=10
SAVE_FREQ=10
BATCH_SIZE=2
BETA=0.1
GRAD_ACC_STEP=20
adapt_rm_dir=ckpts/meta_dpo/adaptation_dpo_gemma_${target_lang}/

CUDA_VISIBLE_DEVICES=0 python adaptation_dpo.py \
    --model_path=$policy_path \
    --batch_size=${BATCH_SIZE} \
    --data_path=$data_path \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=1000 \
    --train_dataset_size=100 \
    --eval_freq=${EVAL_FREQ} \
    --output_dir=$adapt_rm_dir \
    --beta=${BETA} \
    --gradient_accumulation_steps=${GRAD_ACC_STEP} \