target_lang=ro
# -----------------------MAML-SFT-Gemma-----------------------
# Policy model
model_path=google/gemma-3-270m

LEARNING_RATE=5e-4
INNER_LR_COEF=3
NUM_STEPS=3000
EVAL_FREQ=150
SAVE_FREQ=600
BATCH_SIZE=2
INNER_BATCH=4
sft_output_dir=ckpts/meta_rlhf/maml_sft_gemma_${target_lang}
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



# -----------------------Adaptation-SFT-----------------------
model_path=ckpts/meta_rlhf/maml_sft_gemma_${target_lang}
model_path=$(find ${model_path} -type d | sort | tail -n 1)
echo "[SFT] Using ${model_path} to search checkpoint file"

LEARNING_RATE=1e-4
NUM_STEPS=3000
EVAL_FREQ=150
SAVE_FREQ=300
BATCH_SIZE=4
sft_output_dir=ckpts/meta_rlhf/adapted_sft_gemma_${target_lang}
data_path=datasets/multilingual-alpaca-52k/${target_lang}.json

CUDA_VISIBLE_DEVICES="0" python adaptation_sft.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --wandb_name=${WANDB_NAME} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_dataset_size=1000 \
        --save_limit=2 \

# -----------------------GRPO-----------------------

adapt_rm_dir=ckpts/meta_rlhf/adapted_rm_bloom1b7_${target_lang}
REWARD_SEARCH_CKPT=$(find ${adapt_rm_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
reward_path=${REWARD_SEARCH_CKPT}

policy_output_dir=ckpts/meta_rlhf/adapted_sft_gemma_${target_lang}
SFT_SEARCH_CKPT=$(find ${policy_output_dir} -type d | sort | tail -n 1)
echo "[SFT] Using ${SFT_SEARCH_CKPT} to search checkpoint file"
policy_path=${SFT_SEARCH_CKPT}

policy_data_path=datasets/multilingual-rl-tuning-64k/${target_lang}.json
LEARNING_RATE=5e-6
NUM_STEPS=1000
EVAL_FREQ=100
SAVE_FREQ=100
BATCH_SIZE=4
NUM_GENERATIONS=4
PROMPT_LEN=128
SEQ_LEN=512
ACCUMULATION=2
adapt_rm_dir=ckpts/meta_rlhf/rlhf_final_policy_gemma_${target_lang}

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
