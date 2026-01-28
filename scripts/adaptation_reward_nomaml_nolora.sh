lang=$1

LEARNING_RATE=1e-6
NUM_STEPS=10000
EVAL_FREQ=1000
BATCH_SIZE=8
NUM_WARMUP_STEPS=50
reward_path=bigscience/bloom-1b7
reward_output_dir=ckpts/adaption_reward/rm_debug_nolora_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json

# CUDA_VISIBLE_DEVICES=0 python debug_reward.py \
CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
        --reward_path=$reward_path \
        --reward_output_dir=$reward_output_dir \
        --reward_data_path=$reward_data_path \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=${NUM_WARMUP_STEPS}
        

LEARNING_RATE=1e-5
NUM_STEPS=10000
EVAL_FREQ=1000
BATCH_SIZE=8
NUM_WARMUP_STEPS=50
reward_path=bigscience/bloom-1b7
reward_output_dir=ckpts/adaption_reward/rm_debug_nolora_${lang}_lr${LEARNING_RATE}_${NUM_STEPS}steps_batch${BATCH_SIZE}
reward_data_path=datasets/multilingual-ranking-data-42k/$lang.json

# CUDA_VISIBLE_DEVICES=0 python debug_reward.py \
CUDA_VISIBLE_DEVICES=0 python adaptation_reward.py \
        --reward_path=$reward_path \
        --reward_output_dir=$reward_output_dir \
        --reward_data_path=$reward_data_path \
        --learning_rate=${LEARNING_RATE} \
        --max_steps=${NUM_STEPS} \
        --eval_freq=${EVAL_FREQ} \
        --batch_size=${BATCH_SIZE} \
        --num_warmup_steps=${NUM_WARMUP_STEPS}