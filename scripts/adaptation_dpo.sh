lang=$1
model_path=ckpts/rlhf/
output_dir=ckpts/adaption_dpo/policy_$lang
data_path=datasets/multilingual-ranking-data-42k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaptation_dpo.py \
        --model_path=$model_path \
        --output_dir=$output_dir \
        --data_path=$data_path \