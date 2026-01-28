export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=maml-rlhf

lang=ro
model_path=google/gemma-3-270m
rm_model_path=bigscience/bloom-1b7

# -----------------------MAML-SFT for policy Gemma-----------------------
# Policy model

LEARNING_RATE=5e-4
INNER_LR_COEF=3
NUM_STEPS=3000
EVAL_FREQ=150
SAVE_FREQ=600
BATCH_SIZE=2
INNER_BATCH=4
WANDB_NAME=meta_sft_gemma_${lang}_lr${LEARNING_RATE}_inner${INNER_LR_COEF}_${NUM_STEPS}steps_innerbatch${INNER_BATCH}_outerbatch${BATCH_SIZE}
sft_output_dir=ckpts/meta_rlhf_gemma/${WANDB_NAME}
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
model_path=ckpts/meta_rlhf_gemma/maml_sft_gemma_${lang}
model_path=$(find ${model_path} -type d | sort | tail -n 1)
echo "[SFT] Using ${model_path} to search checkpoint file"

LEARNING_RATE=1e-4
NUM_STEPS=3000
EVAL_FREQ=150
SAVE_FREQ=300
BATCH_SIZE=4
WANDB_NAME=adaptation_sft_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
policy_output_dir=ckpts/meta_rlhf_gemma/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k/${target_lang}.json

CUDA_VISIBLE_DEVICES="0" python adaptation_sft.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$policy_output_dir \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --wandb_name=${WANDB_NAME} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --save_freq=${SAVE_FREQ} \
        --eval_dataset_size=1000 \
        --save_limit=2 \


# -----------------------MAML-SFT for RM Bloom-1b7-----------------------
LEARNING_RATE=5e-4
INNER_LR_COEF=3
NUM_STEPS=3000
EVAL_FREQ=150
SAVE_FREQ=600
BATCH_SIZE=2
INNER_BATCH=4
WANDB_NAME=meta_sft_bloom1b7_${lang}_lr${LEARNING_RATE}_inner${INNER_LR_COEF}_${NUM_STEPS}steps_innerbatch${INNER_BATCH}_outerbatch${BATCH_SIZE}
rm_sft_output_dir=ckpts/meta_rlhf_gemma/${WANDB_NAME}
data_path=datasets/multilingual-alpaca-52k

# MODIFY language_list if target language changes
CUDA_VISIBLE_DEVICES="0" python\
        meta_supervised_finetuning.py \
        --use_same_languages_for_eval \
        --language_list ca fr it es \
        --model_path=$rm_model_path \
        --data_path=$data_path \
        --output_dir=$rm_sft_output_dir \
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

adapter_path=$(find ${rm_sft_output_dir} -type d | sort | tail -n 1)

CUDA_VISIBLE_DEVICES=6 python merge_adapter.py \
  --base_model=${reward_model_path} \
  --adapter_path=${adapter_path} \
  --output_path=${rm_sft_output_dir}


# -----------------------MAML-Reward Modelling-----------------------

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
WANDB_NAME=meta_rm_bloom1b7_${lang}_lr${LEARNING_RATE}_inner${INNER_LR_COEF}_${NUM_STEPS}steps_innerbatch${INNER_BATCH}_outerbatch${BATCH_SIZE}
rm_output_dir=ckpts/meta_rlhf_gemma/${WANDB_NAME}


CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --use_same_languages_for_eval \
    --language_list ca fr it es \
    --num_tasks_per_batch ${NUM_TASKS_PER_BATCH} \
    --inner_train_batch_size ${BATCH_SIZE} \
    --micro_batch_size=${OUTER_BATCH_SIZE} \
    --model_path=$rm_sft_output_dir \
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
REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
reward_path=${REWARD_SEARCH_CKPT}


data_path=datasets/multilingual-ranking-data-42k/${lang}.json
LEARNING_RATE=1e-4
NUM_STEPS=300
EVAL_FREQ=30
SAVE_FREQ=30
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
ACCUMULATION=5
WANDB_NAME=adaptation_rm_bloom1b7_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
rm_output_dir=ckpts/meta_rlhf_gemma/${WANDB_NAME}

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



# -----------------------GRPO-----------------------

REWARD_SEARCH_CKPT=$(find ${rm_output_dir} -type d | sort | tail -n 1)
echo "[Reward] Using ${REWARD_SEARCH_CKPT} to search checkpoint file"
reward_path=${REWARD_SEARCH_CKPT}

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
WANDB_NAME=meta_rl_gemma_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}_acc${ACCUMULATION}
policy_output_dir=ckpts/meta_rlhf_gemma/${WANDB_NAME}

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
    --policy_output_dir=${policy_output_dir} \
    --gradient_accumulation_steps=${ACCUMULATION} \
    --reward_use_lora

#------------Evaluation------------
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