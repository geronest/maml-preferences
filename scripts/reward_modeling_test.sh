model_path=bigscience/bloom-1b7
sft_model_path=ckpts/sft_models/sft_model
data_path=datasets/multilingual-ranking-data-42k/
rm_output_dir=ckpts/reward_models/rm

CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --model_path=$model_path \
    --tokenizer_path=$model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=5e-4 \
    --inner_lr=5e-4 \
    --max_steps=3000 \
    --eval_freq=100 \
    --eval_steps=100
