sft_lora_adapter_path=ckpts/masft_models/masft_model_test_loss/checkpoint-3000
reward_lora_adapter_path=ckpts/reward_models/rm/checkpoint-3000
data_path=datasets/multilingual-rl-tuning-64k/
rl_output_dir=ckpts/rlhf/

CUDA_VISIBLE_DEVICES=0 python rl_training.py \
    --sft_lora_adapter_path=$sft_lora_adapter_path\
    --reward_lora_adapter_path=$reward_lora_adapter_path\
    --data_path=$data_path \
    --output_dir=$rl_output_dir\
    --batch_size 2\
    --max_steps=3000 \
    --eval_freq=500 \
    --save_freq=300
