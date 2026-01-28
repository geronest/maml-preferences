model_path=bigscience/bloom-1b7
sft_output_dir=ckpts/sft_models/sft_model_test_loss
data_path=datasets/multilingual-alpaca-52k

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
        supervised_finetuning.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir \
        --batch_size=1 \
        --num_warmup_steps=5 \
        --learning_rate=5e-6 \
        --inner_lr=5e-6 \
        --max_steps=3000 \
        --eval_freq=100 \
        --eval_steps=100
