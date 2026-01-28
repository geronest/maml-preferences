model_path=bigscience/bloom-1b7
sft_output_dir=ckpts/masft_models/masft_model_nolora
data_path=datasets/multilingual-alpaca-52k

# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
#         meta_supervised_finetuning.py \
#         --language_list ro ca it es \
#         --model_path=$model_path \
#         --data_path=$data_path \
#         --output_dir=$sft_output_dir \
#         --batch_size=1 \
#         --num_warmup_steps=5 \
#         --learning_rate=1e-5 \
#         --inner_lr=1e-5 \
#         --num_tasks_per_batch 2\
#         --max_steps=100 \
#         --eval_freq=10 \
#         --eval_steps=10

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
        meta_supervised_finetuning.py \
        --use_same_languages_for_eval \
        --language_list ro ca it es \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=1 \
        --num_warmup_steps=5 \
        --learning_rate=1e-5 \
        --inner_lr=1e-5 \
        --num_tasks_per_batch 2\
        --max_steps=3000 \
        --eval_freq=100 \
        --eval_steps=100
