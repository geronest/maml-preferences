lang=$1

reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

LEARNING_RATE=1e-6
NUM_STEPS=1000
EVAL_FREQ=500
PROMPT_LEN=128
SEQ_LEN=128
BATCH_SIZE=16
NUM_GENERATIONS=16

# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/reward_model_nolora_fr_3000/checkpoint-3000
# policy_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_model_nolora_fr_3000/checkpoint-3000
# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/reward_model_nolora_fr_3000/checkpoint-3000

# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/rm_nolora_fr_lr_steps_batch/checkpoint-5000
# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/rm_nolora_fr_lr1e-6_10steps_batch8/checkpoint-10
reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/rm_debug_nolora_fr_lr1e-5_10000steps_batch8/checkpoint-10000

# policy_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_model_nolora_fr_3000/checkpoint-3000
policy_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_nolora_fr_lr1e-6_10000steps_batch16/checkpoint-10000

policy_output_dir=ckpts/adaption_grpo/policy_nolora_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}

# CUDA_VISIBLE_DEVICES=0 python debug_grpo.py \
CUDA_VISIBLE_DEVICES=0 python adaptation_grpo.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --policy_output_dir=$policy_output_dir \
        --policy_data_path=$policy_data_path \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --prompt_length=${PROMPT_LEN} \
        --seq_length=${SEQ_LEN} \
        --bf16
