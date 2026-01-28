model_path=bigscience/bloom-1b7
data_path=datasets/multilingual-ranking-data-42k/
rm_output_dir=ckpts/reward_models/rm_nolora2
# rm_output_dir=ckpts/reward_models/rm-zerotrain

CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --use_same_languages_for_eval \
    --language_list ro ca it es \
    --num_tasks_per_batch 2 \
    --inner_train_batch_size 2 \
    --model_path=$model_path \
    --tokenizer_path=$model_path \
    --data_path=$data_path \
    --output_dir=$rm_output_dir \
    --num_warmup_steps=5 \
    --learning_rate=1e-5 \
    --inner_lr=1e-5 \
    --max_steps=3000 \
    --eval_freq=100 \
    --eval_steps=100

# CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
#     --model_path=$model_path \
#     --tokenizer_path=$model_path \
#     --data_path=$data_path \
#     --output_dir=$rm_output_dir \
#     --num_warmup_steps=5 \
#     --learning_rate=5e-4 \
#     --inner_lr=5e-4 \
#     --max_steps=1 \
#     --eval_freq=100 \
#     --eval_steps=100
