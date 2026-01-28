lang=$1
# model_path=ckpts/rlhf/
LEARNING_RATE=1e-5
NUM_STEPS=1000
EVAL_FREQ=500
GRAD_ACC_STEPS=16
BATCH_SIZE=4
# model_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_model_nolora_fr_3000/checkpoint-3000
model_path=/home/ubuntu/virginia0/MAML-Okapi/MAML_Okapi/ckpts/sft_models/sft_nolora_fr_lr1e-6_10000steps_batch16/checkpoint-10000
output_dir=ckpts/dpo_nolora_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}/policy_$lang
data_path=datasets/multilingual-ranking-data-42k/$lang.json

CUDA_VISIBLE_DEVICES=0 python adaptation_dpo.py \
        --model_path=$model_path \
        --output_dir=$output_dir \
        --data_path=$data_path \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --batch_size=${BATCH_SIZE} \
        --gradient_accumulation_steps=${GRAD_ACC_STEPS} \
        --eval_batch_size=16 \
        --bf16