#!bin/bash

export WANDB_API_KEY=b31f5ed0767abb05596b308c3c5efc4384086962
export WANDB_PROJECT=maml-rlhf

# Common Parameters
model_path=bigscience/bloom-7b1
reward_model_path=bigscience/bloom-7b1
lang=ca

GPU_ALLOC=0

# SFT
INNER_LR_COEF=3
LEARNING_RATE=3e-5
NUM_STEPS=100
EVAL_FREQ=100
SAVE_FREQ=100
INNER_BATCH_SIZE=2
NUM_INNER_STEPS=1
GRAD_ACC=40
WANDB_NAME=bloom7b1_lora_oldfometasft_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${INNER_BATCH_SIZE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC}
META_NUM_STEPS=${NUM_STEPS}
META_GRAD_ACC=${GRAD_ACC}

sft_output_dir=ckpts/maml-sft/${WANDB_NAME}
merged_sft_output_dir=ckpts/maml-sft/merged_${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python meta_supervised_finetuning.py \
        --use_lora \
        --use_same_languages_for_eval \
        --language_list ro fr it es \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --inner_train_batch_size=${INNER_BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --inner_lr_coef=${INNER_LR_COEF} \
        --wandb_name=${WANDB_NAME} \
        --num_tasks_per_batch 2 \
        --num_inner_steps ${NUM_INNER_STEPS} \
        --gradient_accumulation_steps ${GRAD_ACC} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_steps=100

adapter_path=$(find ${sft_output_dir} -type d | sort | tail -n 1)
CUDA_VISIBLE_DEVICES=$GPU_ALLOC python merge_adapter.py \
  --base_model=${model_path} \
  --adapter_path=${adapter_path} \
  --output_path=${merged_sft_output_dir}

# Reward Modeling
data_path=datasets/multilingual-ranking-data-42k/
reward_path_load=$merged_sft_output_dir
LEARNING_RATE=3e-5
INNER_LR_COEF=3
NUM_INNER_STEPS=1
NUM_STEPS=400
EVAL_FREQ=100
SAVE_FREQ=400
NUM_TASKS_PER_BATCH=2
BATCH_SIZE=2
GRAD_ACC_STEPS=40
LR_SCHEDULER_TYPE=constant
WANDB_NAME=bloom7b1mtsft${META_NUM_STEPS}_lora__${lang}_oldfometarm_4langs_${NUM_TASKS_PER_BATCH}tasks_${NUM_STEPS}steps_lr${LR_SCHEDULER_TYPE}${LEARNING_RATE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC_STEPS}
rm_output_dir=ckpts/maml-rm/${WANDB_NAME}
META_RM_STEPS=${NUM_STEPS}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python reward_modeling.py \
    --use_lora \
    --use_same_languages_for_eval \
    --language_list ro fr it es \
    --num_tasks_per_batch ${NUM_TASKS_PER_BATCH} \
    --inner_train_batch_size ${BATCH_SIZE} \
    --model_path=$reward_path_load \
    --tokenizer_path=$reward_model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --inner_lr_coef=${INNER_LR_COEF} \
    --lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --eval_steps=100

# Reward Modeling - Adapatation
reward_path=$rm_output_dir
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
META_LEARNING_RATE=$LEARNING_RATE
LEARNING_RATE=1e-5
NUM_STEPS=100
EVAL_FREQ=50
SAVE_FREQ=${NUM_STEPS}
BATCH_SIZE=40
TRAIN_DATA_SIZE=100
GRAD_ACC_STEPS=1
WEIGHT_DECAY=1e-6
LR_SCHEDULER_TYPE=constant
WANDB_NAME=${lang}_bloom7b1_lora_oldadaptrm_mlr${META_LEARNING_RATE}mrm${META_RM_STEPS}_${NUM_INNER_STEPS}msteps_4langs_ga${META_GRAD_ACC}_lr${LR_SCHEDULER_TYPE}${LEARNING_RATE}_gradacc${GRAD_ACC_STEPS}_${NUM_STEPS}steps_batch${BATCH_SIZE}_decay${WEIGHT_DECAY}_tdata${TRAIN_DATA_SIZE}
rm_output_dir=ckpts/adaptation-rm/${WANDB_NAME}
RM_NUM_STEPS=${NUM_STEPS}
RM_GRAD_ACC=${GRAD_ACC_STEPS}
RM_LEARNING_RATE=${LEARNING_RATE}

CUDA_VISIBLE_DEVICES=${GPU_ALLOC} python adaptation_reward.py \
    --reward_path=$reward_path \
    --reward_output_dir=$rm_output_dir \
    --reward_data_path=$reward_data_path \
    --train_dataset_size=${TRAIN_DATA_SIZE} \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=1000 \
    --weight_decay=${WEIGHT_DECAY} \
    --wandb_name=${WANDB_NAME} \
    --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
    --batch_size=${BATCH_SIZE} \
    --eval_steps=100 \
    --lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    --use_lora

# Adaptation-SFT
sft_path=bigscience/bloom-7b1

LEARNING_RATE=1e-4
TRAIN_DATA_SIZE=100
NUM_STEPS=100
EVAL_FREQ=10
SAVE_FREQ=100
BATCH_SIZE=40
GRAD_ACC_STEPS=1
WANDB_NAME=${lang}_bloom7b1_oldrawsft_tdata${TRAIN_DATA_SIZE}_mtsft${META_NUM_STEPS}_lora_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${GRAD_ACC_STEPS}
SFT_NUM_STEPS=${NUM_STEPS}

sft_output_dir=ckpts/adaptation-sft/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k/${lang}.json

LANGUAGE_LIST="${lang}"
CUDA_VISIBLE_DEVICES=${GPU_ALLOC} python supervised_finetuning.py \
        --model_path=$sft_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --train_dataset_size=${TRAIN_DATA_SIZE} \
        --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
        --wandb_name=${WANDB_NAME} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --warmup_ratio=0.05 \
        --use_lora 


# GRPO
mpolicy_output_dir=$sft_output_dir
REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
MP_SEARCH_CKPT=$(find ${mpolicy_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[Adaptation:Meta-policy] Using ${MP_SEARCH_CKPT} to search checkpoint file"

reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json
TRAIN_DATA_SIZE=100
LEARNING_RATE=1e-4
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
WANDB_NAME=${lang}_bloom7b1_oldadaptgrpo_masft${META_NUM_STEPS}_lora_rawsft${SFT_NUM_STEPS}mrm${RM_NUM_STEPS}_${NUM_INNER_STEPS}msteps_rlr${RM_LEARNING_RATE}_tdata${TRAIN_DATA_SIZE}4langs_ga${RM_GRAD_ACC}_lr${LR_SCHEDULER_TYPE}${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_gradacc${GRAD_ACC_STEPS}_decay${WEIGHT_DECAY}

reward_path=${REWARD_SEARCH_CKPT}
mpolicy_path=${MP_SEARCH_CKPT}

policy_output_dir=ckpts/adaption_grpo/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python adaptation_grpo.py \
        --reward_path=$reward_path \
        --policy_path=$mpolicy_path \
        --policy_output_dir=$policy_output_dir \
        --policy_data_path=$policy_data_path \
        --train_dataset_size=${TRAIN_DATA_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --weight_decay=${WEIGHT_DECAY} \
        --batch_size=${BATCH_SIZE} \
        --wandb_name=${WANDB_NAME} \
        --num_generations=${NUM_GENERATIONS} \
        --prompt_length=${PROMPT_LEN} \
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