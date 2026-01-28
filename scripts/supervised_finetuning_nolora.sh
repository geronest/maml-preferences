model_path=bigscience/bloom-1b7
# sft_output_dir=ckpts/sft_models/sft_model_3000

LEARNING_RATE=2e-5
NUM_STEPS=1000
EVAL_FREQ=250
BATCH_SIZE=128
sft_output_dir=ckpts/sft_models/sft_nolora_fr_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
data_path=datasets/multilingual-alpaca-52k

LANGUAGE_LIST="fr"
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
        supervised_finetuning.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=5 \
        --learning_rate=${LEARNING_RATE} \
        --inner_lr=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --eval_steps=100 \
        --num_tasks_per_batch 1 \
        --language_list ${LANGUAGE_LIST}
