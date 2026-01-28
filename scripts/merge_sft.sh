base_model=bigscience/bloom-7b1
input_path=$1
sft_adapter_path=ckpts/sft_models/${input_path}
adapter_path=$(find ${sft_adapter_path} -type d | sort | tail -n 1)
output_path=ckpts/sft_models/merged_${input_path}

CUDA_VISIBLE_DEVICES=6 python merge_adapter.py \
  --base_model=${base_model} \
  --adapter_path=${adapter_path} \
  --output_path=${output_path}

