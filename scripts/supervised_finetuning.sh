model_path=bigscience/bloom-1b7
sft_output_dir=ckpts/sft_models/sft_model
data_path=datasets/multilingual-alpaca-52k

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 \
        supervised_finetuning.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir\
