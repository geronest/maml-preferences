
sft_lora_adapter_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/masft_models/masft_model_nolora/checkpoint-3000
# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/reward_models/rm_nolora/checkpoint-3000
reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/reward_models/rm_nolora2/checkpoint-3000
data_path=datasets/multilingual-rl-tuning-64k/
rl_output_dir=ckpts/meta_rlhf_nolora/

# CUDA_VISIBLE_DEVICES=0 python rl_training.py \
#     --sft_lora_adapter_path=$sft_lora_adapter_path\
#     --reward_path=$reward_path\
#     --data_path=$data_path \
#     --batch_size 2 \
#     --max_steps=3000 \
#     --eval_freq=100 \
#     --eval_steps=10 \
#     --output_dir=$rl_output_dir

rl_output_dir=ckpts/meta_rlhf_nolora_100step/
CUDA_VISIBLE_DEVICES=0 python rl_training-debug.py \
    --sft_lora_adapter_path=$sft_lora_adapter_path\
    --reward_path=$reward_path\
    --data_path=$data_path \
    --batch_size 4 \
    --max_steps=100 \
    --eval_freq=50 \
    --eval_steps=10 \
    --output_dir=$rl_output_dir


# sft_lora_adapter_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/masft_models/masft_model_test_loss/checkpoint-3000
# reward_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/reward_models/rm/checkpoint-3000
# data_path=datasets/multilingual-rl-tuning-64k/
# rl_output_dir=ckpts/meta_rlhf_lora/

# CUDA_VISIBLE_DEVICES=0 python rl_training.py \
#     --sft_lora_adapter_path=$sft_lora_adapter_path\
#     --reward_path=$reward_path\
#     --data_path=$data_path \
#     --batch_size 2 \
#     --max_steps=3000 \
#     --eval_freq=100 \
#     --eval_steps=100 \
#     --output_dir=$rl_output_dir \
#     --use_lora
