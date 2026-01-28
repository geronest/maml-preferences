#!bin/bash

export WANDB_API_KEY=4463d49cb6d35aa2dbab0fe629f9aaeefc8d8898
export WANDB_PROJECT=meta-rlhf-gemma_dpo

# Common Parameters
# model_path=google/gemma-3-270m

# adapt_rm_dir=ckpts/adapt_rm/adaptrm_bloom_fr_3000steps_lr5e-5_batch4_bestshort
# REWARD_SEARCH_CKPT=$(find ${adapt_rm_dir} -type d | sort | tail -n 1)
# echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
# reward_path=${REWARD_SEARCH_CKPT}

policy_output_dir=ckpts/sft_models/sft_gemma_fr_lr5e-5_inner0.1_3000steps_innerBatch4_batch2
# policy_output_dir=ckpts/sft_models/sft_pipe2_bloom_fr_lr5e-4_inner0.01_3000steps_innerBatch4_batch2
SFT_SEARCH_CKPT=$(find ${policy_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
policy_path=${SFT_SEARCH_CKPT}

data_path=datasets/multilingual-ranking-data-42k/
# reward_data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=$1
INNER_LR_COEF=$2
NUM_STEPS=3000
EVAL_FREQ=300
SAVE_FREQ=600
BATCH_SIZE=2
EVAl_BATCH_SIZE=2
NUM_TASKS_PER_BATCH=2
# WANDB_NAME=dpo_bloom_fr_${NUM_STEPS}steps_lr${LEARNING_RATE}_batch${BATCH_SIZE}
WANDB_NAME=dpo_gemma_fr_${NUM_STEPS}steps_lr${LEARNING_RATE}_batch${BATCH_SIZE}
adapt_rm_dir=ckpts/meta_dpo/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python dpo.py \
    --use_same_languages_for_eval \
    --model_path=$policy_path \
    --inner_train_batch_size=${BATCH_SIZE} \
    --eval_batch_size=${EVAl_BATCH_SIZE} \
    --data_path=$data_path \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=100 \
    --output_dir=${adapt_rm_dir} \
    --eval_freq=${EVAL_FREQ} \
    --inner_lr_coef=${INNER_LR_COEF} \
    --num_tasks_per_batch=${NUM_TASKS_PER_BATCH} \
    # --use_lora \
