
target_lang=ro
# -----------------------MAML-SFT-Bloom-----------------------
model_path=bigscience/bloom-1b7

LEARNING_RATE=5e-4
INNER_LR_COEF=3
NUM_STEPS=3000
EVAL_FREQ=150
SAVE_FREQ=600
BATCH_SIZE=2
INNER_BATCH=4
sft_output_dir=ckpts/meta_rlhf/maml_sft_bloom1b7_${target_lang}
data_path=datasets/multilingual-alpaca-52k

# MODIFY language_list if target language changes
CUDA_VISIBLE_DEVICES="0" python\
        meta_supervised_finetuning.py \
        --use_same_languages_for_eval \
        --language_list ca fr it es \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --inner_train_batch_size=${INNER_BATCH} \
        --micro_batch_size=${BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --inner_lr_coef=${INNER_LR_COEF} \
        --wandb_name=${WANDB_NAME} \
        --num_tasks_per_batch 2 \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_steps=1000 \


# -----------------------MAML-Reward Modelling-----------------------

# Common Parameters
sft_model_path=ckpts/meta_rlhf/maml_sft_bloom1b7_${target_lang}
reward_model_path=$(find ${sft_model_path} -type d | sort | tail -n 1)
echo "[RM] Using ${reward_model_path} to search checkpoint file"

data_path=datasets/multilingual-ranking-data-42k/
LEARNING_RATE=5e-4
INNER_LR_COEF=3
NUM_STEPS=300
EVAL_FREQ=30
SAVE_FREQ=30
NUM_TASKS_PER_BATCH=2
BATCH_SIZE=4
ACCUMULATION=40
OUTER_BATCH_SIZE=2
WEIGHT_DECAY=1e-6
rm_output_dir=ckpts/meta_rlhf/maml_rm_bloom1b7_${target_lang}


CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --use_same_languages_for_eval \
    --language_list ca fr it es \
    --num_tasks_per_batch ${NUM_TASKS_PER_BATCH} \
    --inner_train_batch_size ${BATCH_SIZE} \
    --micro_batch_size=${OUTER_BATCH_SIZE} \
    --model_path=$model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --inner_lr_coef=${INNER_LR_COEF} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --eval_freq=${EVAL_FREQ} \
    --save_freq=${SAVE_FREQ} \
    --eval_steps=1000 \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --weight_decay=${WEIGHT_DECAY} \
    --use_lora

# -----------------------Adaptation-Reward Modelling-----------------------
maml_rm_dir=ckpts/meta_rlhf/maml_rm_bloom1b7_${target_lang}
REWARD_SEARCH_CKPT=$(find ${maml_rm_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
reward_path=${REWARD_SEARCH_CKPT}


data_path=datasets/multilingual-ranking-data-42k/${target_lang}.json
LEARNING_RATE=1e-4
NUM_STEPS=300
EVAL_FREQ=30
SAVE_FREQ=30
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
ACCUMULATION=5
rm_output_dir=ckpts/meta_rlhf/adapted_rm_bloom1b7_${target_lang}

CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
    --batch_size=${BATCH_SIZE} \
    --reward_path=$reward_path \
    --reward_data_path=$data_path \
    --reward_output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_freq=${EVAL_FREQ} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --use_lora \
    --eval_dataset_size=1000 \


# -----------------------Adaptation-GRPO-----------------------

lang=ca

adapt_rm_dir=ckpts/adapt_rm/adaptrm_bloom_ca_metabatch40_300steps_acc5_lr5e-4_batch8_3e-4-1
REWARD_SEARCH_CKPT=$(find ${adapt_rm_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
reward_path=${REWARD_SEARCH_CKPT}

policy_output_dir=ckpts/adapt_sft/adaptsft_gemma_ca_lr5e-6_3000steps_batch4_full
SFT_SEARCH_CKPT=$(find ${policy_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
policy_path=${SFT_SEARCH_CKPT}

policy_data_path=datasets/multilingual-rl-tuning-64k/${lang}.json
LEARNING_RATE=1e-5
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=50
BATCH_SIZE=4
NUM_GENERATIONS=4
PROMPT_LEN=128
SEQ_LEN=512
ACCUMULATION=2
WANDB_NAME=adaptrl_${lang}_${NUM_STEPS}steps_acc${ACCUMULATION}_lr${LEARNING_RATE}_batch${BATCH_SIZE}
adapt_rm_dir=ckpts/adapt_rl/${WANDB_NAME}

CUDA_VISIBLE_DEVICES=0 python adaptation_grpo.py \
    --policy_path=$policy_path \
    --batch_size=${BATCH_SIZE} \
    --reward_path=$reward_path \
    --policy_data_path=$policy_data_path \
    --num_warmup_steps=5 \
    --learning_rate=${LEARNING_RATE} \
    --wandb_name=${WANDB_NAME} \
    --max_steps=${NUM_STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_dataset_size=100 \
    --eval_freq=${EVAL_FREQ} \
    --num_generations=${NUM_GENERATIONS} \
    --prompt_length=${PROMPT_LEN} \
    --seq_length=${SEQ_LEN} \
    --policy_output_dir=${adapt_rm_dir} \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --reward_use_lora
