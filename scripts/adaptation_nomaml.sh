lang=$1

# reward_path=ckpts/reward_models/checkpoint-500
# reward_path=bigscience/bloom-7b1
# policy_path=ckpts/sft_models/sft_model_test_loss/checkpoint-3000
# reward_path=Meta-Okapi/baseline0_reward_model_kn_checkpoint-50
# policy_path=Meta-Okapi/baseline0_policy_kn_checkpoint-50
# reward_output_dir=ckpts/adaption_reward_testupload/reward_model_$lang
# policy_output_dir=ckpts/adaption_grpo_testupload/policy_$lang
# reward_output_dir=ckpts/adaption_reward_nomaml/reward_model_$lang
# policy_output_dir=ckpts/adaption_reward_nomaml/policy_$lang
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/adaption_reward/reward_model_fr_3000/checkpoint-3000
# policy_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_model_fr_3000/checkpoint-3000

# policy_output_dir=ckpts/adaption_grpo/policy_${lang}_3000
policy_output_dir=ckpts/adaption_grpo/policy_${lang}_debug

# CUDA_VISIBLE_DEVICES=0 python adaption_reward.py \
#         --reward_path=$reward_path \
#         --reward_output_dir=$reward_output_dir \
#         --reward_data_path=$reward_data_path

# CUDA_VISIBLE_DEVICES=0 python adaption_grpo.py \
# CUDA_VISIBLE_DEVICES=0 python debug_grpo.py \
#         --reward_lora_adapter_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_output_dir=$policy_output_dir \
#         --policy_data_path=$policy_data_path \
#         --max_steps 30 \
#         --use_lora \
#         --eval_freq 10 

policy_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_model/checkpoint-5000
LEARNING_RATE=1e-6
NUM_STEPS=1000
EVAL_FREQ=500
PROMPT_LEN=128
SEQ_LEN=128
BATCH_SIZE=16
NUM_GENERATIONS=16
# CUDA_VISIBLE_DEVICES=0 python adaptation_grpo.py \
CUDA_VISIBLE_DEVICES=0 python debug_grpo.py \
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
        --use_lora \
        --bf16
