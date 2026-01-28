lang=$1
train_dataset_size=$2
model_path=ckpts/rlhf/checkpoint-3000
output_dir=ckpts/adaption_dpo/policy_${lang}_${train_dataset_size}
data_path=datasets/multilingual-ranking-data-42k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaption_dpo.py \
        --model_path=$model_path \
        --output_dir=$output_dir \
        --data_path=$data_path \
        --train_dataset_size=$train_dataset_size\
        --batch_size 2\
        --max_steps=3000 \
        --eval_freq=500 \
        --save_freq=300