#!bin/bash

export WANDB_API_KEY=b31f5ed0767abb05596b308c3c5efc4384086962
export WANDB_PROJECT=maml-rlhf

# Common Parameters
model_path=bigscience/bloom-1b7

# SFT
# INNER_LR_COEF=3
# LEARNING_RATE=1e-6
# NUM_STEPS=800
# EVAL_FREQ=100
# SAVE_FREQ=800
# INNER_BATCH_SIZE=2
# GRAD_ACC=4
# WANDB_NAME=fometasft_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${INNER_BATCH_SIZE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC}

# # sft_output_dir=ckpts/sft_models/${WANDB_NAME}
# sft_output_dir=ckpts/sft_models/${WANDB_NAME}
# data_path=datasets/multilingual-alpaca-52k

# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
#         meta_supervised_finetuning.py \
#         --use_same_languages_for_eval \
#         --language_list ro ca it es \
#         --model_path=$model_path \
#         --data_path=$data_path \
#         --output_dir=$sft_output_dir \
#         --inner_train_batch_size=${INNER_BATCH_SIZE} \
#         --num_warmup_steps=5 \
#         --learning_rate=${LEARNING_RATE} \
#         --inner_lr_coef=${INNER_LR_COEF} \
#         --wandb_name=${WANDB_NAME} \
#         --num_tasks_per_batch 2 \
#         --gradient_accumulation_steps ${GRAD_ACC} \
#         --max_steps=${NUM_STEPS} \
#         --eval_freq=${EVAL_FREQ} \
#         --save_freq=${SAVE_FREQ} \
#         --eval_steps=100

INNER_LR_COEF=3
LEARNING_RATE=3e-6
NUM_STEPS=400
EVAL_FREQ=100
SAVE_FREQ=200
INNER_BATCH_SIZE=2
GRAD_ACC=4
WANDB_NAME=fometasft_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${INNER_BATCH_SIZE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC}

# sft_output_dir=ckpts/sft_models/${WANDB_NAME}
sft_output_dir=ckpts/sft_models/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
#         meta_supervised_finetuning.py \
#         --use_same_languages_for_eval \
#         --language_list ro ca it es \
#         --model_path=$model_path \
#         --data_path=$data_path \
#         --output_dir=$sft_output_dir \
#         --inner_train_batch_size=${INNER_BATCH_SIZE} \
#         --num_warmup_steps=5 \
#         --learning_rate=${LEARNING_RATE} \
#         --inner_lr_coef=${INNER_LR_COEF} \
#         --wandb_name=${WANDB_NAME} \
#         --num_tasks_per_batch 2 \
#         --gradient_accumulation_steps ${GRAD_ACC} \
#         --max_steps=${NUM_STEPS} \
#         --eval_freq=${EVAL_FREQ} \
#         --save_freq=${SAVE_FREQ} \
#         --eval_steps=100

# INNER_LR_COEF=3
# LEARNING_RATE=1e-5
# NUM_STEPS=800
# EVAL_FREQ=100
# SAVE_FREQ=800
# INNER_BATCH_SIZE=2
# GRAD_ACC=4
# WANDB_NAME=fometasft_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${INNER_BATCH_SIZE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC}

# # sft_output_dir=ckpts/sft_models/${WANDB_NAME}
# sft_output_dir=ckpts/sft_models/${WANDB_NAME}
# data_path=datasets/multilingual-alpaca-52k

# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
#         meta_supervised_finetuning.py \
#         --use_same_languages_for_eval \
#         --language_list ro ca it es \
#         --model_path=$model_path \
#         --data_path=$data_path \
#         --output_dir=$sft_output_dir \
#         --inner_train_batch_size=${INNER_BATCH_SIZE} \
#         --num_warmup_steps=5 \
#         --learning_rate=${LEARNING_RATE} \
#         --inner_lr_coef=${INNER_LR_COEF} \
#         --wandb_name=${WANDB_NAME} \
#         --num_tasks_per_batch 2 \
#         --gradient_accumulation_steps ${GRAD_ACC} \
#         --max_steps=${NUM_STEPS} \
#         --eval_freq=${EVAL_FREQ} \
#         --save_freq=${SAVE_FREQ} \
#         --eval_steps=100

# Reward Modeling
data_path=datasets/multilingual-ranking-data-42k/

# LEARNING_RATE=1e-5
# INNER_LR_COEF=3
# NUM_STEPS=200
# EVAL_FREQ=50
# SAVE_FREQ=200
# # NUM_STEPS=10
# # EVAL_FREQ=5
# # SAVE_FREQ=20
# NUM_TASKS_PER_BATCH=2
# BATCH_SIZE=2
# GRAD_ACC_STEPS=4
# WANDB_NAME=fometarm_${NUM_TASKS_PER_BATCH}tasks_nolora_${NUM_STEPS}steps_lr${LEARNING_RATE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC_STEPS}
# rm_output_dir=ckpts/reward_models/${WANDB_NAME}

# # --language_list ro ca it es \
# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --use_same_languages_for_eval \
#     --language_list ro ca it es \
#     --num_tasks_per_batch ${NUM_TASKS_PER_BATCH} \
#     --inner_train_batch_size ${BATCH_SIZE} \
#     --model_path=$model_path \
#     --tokenizer_path=$model_path \
#     --data_path=$data_path \
#     --output_dir=$rm_output_dir \
#     --num_warmup_steps=50 \
#     --learning_rate=${LEARNING_RATE} \
#     --inner_lr_coef=${INNER_LR_COEF} \
#     --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
#     --wandb_name=${WANDB_NAME} \
#     --max_steps=${NUM_STEPS} \
#     --eval_freq=${EVAL_FREQ} \
#     --eval_steps=100

LEARNING_RATE=3e-5
INNER_LR_COEF=3
NUM_STEPS=400
EVAL_FREQ=50
SAVE_FREQ=200
# NUM_STEPS=10
# EVAL_FREQ=5
# SAVE_FREQ=20
NUM_TASKS_PER_BATCH=2
BATCH_SIZE=2
GRAD_ACC_STEPS=4
WANDB_NAME=fometarm_${NUM_TASKS_PER_BATCH}tasks_nolora_${NUM_STEPS}steps_lr${LEARNING_RATE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC_STEPS}
rm_output_dir=ckpts/reward_models/${WANDB_NAME}

# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --use_same_languages_for_eval \
#     --language_list ro ca it es \
#     --num_tasks_per_batch ${NUM_TASKS_PER_BATCH} \
#     --inner_train_batch_size ${BATCH_SIZE} \
#     --model_path=$model_path \
#     --tokenizer_path=$model_path \
#     --data_path=$data_path \
#     --output_dir=$rm_output_dir \
#     --num_warmup_steps=50 \
#     --learning_rate=${LEARNING_RATE} \
#     --inner_lr_coef=${INNER_LR_COEF} \
#     --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
#     --wandb_name=${WANDB_NAME} \
#     --max_steps=${NUM_STEPS} \
#     --eval_freq=${EVAL_FREQ} \
#     --eval_steps=100


# # GRPO
REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
SFT_SEARCH_CKPT=$(find ${sft_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"

# lang=fr
reward_data_path=datasets/multilingual-ranking-data-42k/
policy_data_path=datasets/multilingual-rl-tuning-64k/

LEARNING_RATE=1e-5
WEIGHT_DECAY=0.05
NUM_STEPS=400
EVAL_FREQ=100
SAVE_FREQ=200
PROMPT_LEN=128
SEQ_LEN=512
BATCH_SIZE=4
NUM_GENERATIONS=4
GRAD_ACC_STEPS=4
WANDB_NAME=fometagrpo_nolora_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_decay${WEIGHT_DECAY}

reward_path=${REWARD_SEARCH_CKPT}
sft_path=${SFT_SEARCH_CKPT}

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
    --prompt_length=${PROMPT_LEN} \
    --num_rollouts=${NUM_GENERATIONS} \
    --seq_length=${SEQ_LEN} \
    --gradient_accumulation_steps=${GRAD_ACC_STEPS}
