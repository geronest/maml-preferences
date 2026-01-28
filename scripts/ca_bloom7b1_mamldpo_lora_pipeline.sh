#!bin/bash

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=maml-dpo-bloom7b1

# Common Parameters
model_path=bigscience/bloom-7b1
reward_model_path=bigscience/bloom-7b1
lang=ca

GPU_ALLOC=5

# MAML_SFT
INNER_LR_COEF=3
LEARNING_RATE=0
NUM_STEPS=1
EVAL_FREQ=1
SAVE_FREQ=1
INNER_BATCH_SIZE=2
GRAD_ACC=40
WANDB_NAME=bloom7b1_lora_fometasft_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${INNER_BATCH_SIZE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC}
META_NUM_STEPS=${NUM_STEPS}
META_GRAD_ACC=${GRAD_ACC}

sft_output_dir=ckpts/sft_models/${WANDB_NAME}
merged_sft_output_dir=ckpts/sft_models/merged_${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python meta_supervised_finetuning.py \
        --use_lora \
        --use_same_languages_for_eval \
        --language_list ro ca it es \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --inner_train_batch_size=${INNER_BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --inner_lr_coef=${INNER_LR_COEF} \
        --wandb_name=${WANDB_NAME} \
        --num_tasks_per_batch 2 \
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

# MAML-DPO
sft_path=$sft_output_dir
dpo_data_path=datasets/multilingual-ranking-data-42k/

INNER_LR_COEF=3
LEARNING_RATE=3e-5
NUM_STEPS=400
EVAL_FREQ=100
SAVE_FREQ=400
INNER_BATCH_SIZE=6
NUM_INNER_STEPS=2
NUM_TASKS_PER_BATCH=1
GRAD_ACC=10
WANDB_NAME=bloom7b1_fometadpo_lora_msft${META_NUM_STEPS}_${NUM_INNER_STEPS}msteps_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${INNER_BATCH_SIZE}_innerlr${INNER_LR_COEF}_gradacc${GRAD_ACC}
MDPO_NUM_STEPS=${NUM_STEPS}
MDPO_GRAD_ACC=${GRAD_ACC}

mdpo_output_dir=ckpts/maml-dpo/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python maml_dpo.py \
        --use_lora \
        --use_same_languages_for_eval \
        --language_list ro fr it es \
        --model_path=$sft_path \
        --data_path=$dpo_data_path \
        --output_dir=$mdpo_output_dir \
        --inner_train_batch_size=${INNER_BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --inner_lr_coef=${INNER_LR_COEF} \
        --wandb_name=${WANDB_NAME} \
        --eval_dataset_size=1000 \
        --num_tasks_per_batch 2 \
        --gradient_accumulation_steps ${GRAD_ACC} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_steps=100

# Adaptation-DPO
mpolicy_output_dir=$mdpo_output_dir
MP_SEARCH_CKPT=$(find ${mpolicy_output_dir} -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -t $'\t' -k1,1V | tail -n 1 | cut -f2-)
echo "[Adaptation:Meta-policy] Using ${MP_SEARCH_CKPT} to search checkpoint file"

dpo_data_path=datasets/multilingual-ranking-data-42k/$lang.json

LEARNING_RATE=1e-5
LR_SCHEDULER_TYPE=constant
TRAIN_DATA_SIZE=8000
WEIGHT_DECAY=0.05
NUM_STEPS=200
EVAL_FREQ=10
SAVE_FREQ=100
BATCH_SIZE=10
GRAD_ACC_STEPS=4
WANDB_NAME=bloom7b1_adaptdpo_tdata${TRAIN_DATA_SIZE}_lora_msft${META_NUM_STEPS}mdpo${MDPO_NUM_STEPS}_${NUM_INNER_STEPS}msteps_4langs_innerlr${INNER_LR_COEF}_ga${MDPO_GRAD_ACC}_${lang}_lr${LR_SCHEDULER_TYPE}${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_gradacc${GRAD_ACC_STEPS}_decay${WEIGHT_DECAY}

mpolicy_path=${MP_SEARCH_CKPT}

policy_output_dir=ckpts/adaptation_dpo/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=$GPU_ALLOC python adaptation_dpo.py \
        --model_path=$mpolicy_path \
        --output_dir=$policy_output_dir \
        --data_path=$dpo_data_path \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --batch_size=${BATCH_SIZE} \
        --weight_decay=${WEIGHT_DECAY} \
        --train_dataset_size=${TRAIN_DATA_SIZE} \
        --eval_dataset_size=1000 \
        --wandb_name=${WANDB_NAME} \
        --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
        --bf16 \
        --use_lora \
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