#!bin/bash

lang=fr
policy_data_path=datasets/multilingual-rl-tuning-64k/$lang.json

BATCH_SIZE=16
PROMPT_LEN=128
SEQ_LEN=512
EVAL_SIZE=1000

lang=fr
reward_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/reward_models/bloom7b1_judgerm_decay1e-6_fr_lr5e-5_10000steps_batch16_acc1/checkpoint-10000
policy_path=bigscience/bloom-7b1

RESULT_NAME=${lang}_judgerm_bloom7b1-base

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16


# lang=fr
# policy_path=ckpts/adaption_dpo/adaptdpo_lora_msft100mdpo1004langs_innerlr0_ga40_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}_adaptdpo_lora_msft100mdpo1004langs_innerlr0_ga40_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=3 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=ckpts/adaption_dpo/adaptdpo_lora_msft100mdpo1004langs_innerlr0_ga40_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}_adaptdpo_lora_msft100mdpo1004langs_innerlr0_ga40_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=3 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=ckpts/adaption_grpo/bloom7b1_adaptgrpo_lora_adaptsft100rm2004langs_ga1_fr_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}_bloom7b1_adaptgrpo_lora_adaptsft100rm2004langs_ga1_fr_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16


# lang=fr
# policy_path=ckpts/adaption_dpo/dpo_lora_sft1000_fr_lrconstant1e-5_1000steps_batch10_gradacc4_decay0.05/checkpoint-1000
# RESULT_NAME=${lang}_dpo_lora_sft1000_fr_lrconstant1e-5_1000steps_batch10_gradacc4_decay0.05

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=ckpts/adaption_grpo/bloom7b1_lora_sft1000lr1e-4_sftrm1000batch40_grpo_fr_lrconstant1e-4_1000steps_batch4_decay0.05/checkpoint-1000
# RESULT_NAME=${lang}_bloom7b1_lora_sft1000lr1e-4_sftrm1000batch40_grpo_fr_lrconstant1e-4_1000steps_batch4_decay0.05

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
# #         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/fr/bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-5_tdata1004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptgrpo_tdata100_lora_adaptsft100mrm200_5msteps_rlr1e-5_tdata1004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/fr/bloom7b1_lora_sft100lr1e-4_sftrm200rlr1e-5batch100tdata100_grpo_fr_lrconstant1e-5_500steps_batch4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_grpo_tdata100_lora_sft100lr1e-4_sftrm200rlr1e-5batch100tdata100_grpo_fr_lrconstant1e-5_500steps_batch4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/dpo_lora_sft100_tdata100all_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/n_${lang}_dpo_tdata100_lora_sft100_tdata100all_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_lora_msft10mdpo4004langs_innerlr3_ga40_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata100_lora_msft10mdpo4004langs_innerlr3_ga40_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata1000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata1000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata1000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata1000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata8000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata8000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/dpo_lora_sft1000_tdata10000all_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05/checkpoint-2000
# RESULT_NAME=${lang}/n_${lang}_dpo_lora_sft1000_tdata10000all_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05_2000steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata10000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05/checkpoint-2000
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata10000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05_2000steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/multitask_grpo/bloom7b1_multitask-grpo_nomasft_lora_tdata8000_adaptsftrm2004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_multitask-grpo_nomasft_lora_tdata8000_adaptsftrm2004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-4_tdata80004langs_ga1_fr_lrconstant1e-4_500steps_batch4_gradacc4_decay0.05/checkpoint-500
# RESULT_NAME=${lang}/n_${lang}_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-4_tdata80004langs_ga1_fr_lrconstant1e-4_500steps_batch4_gradacc4_decay0.05_500steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_adaptgrpo_nomasft_lora_adaptsft1000mrm1000_5msteps_rlr1e-5_tdata400004langs_ga1_ro_lrconstant1e-5_1000steps_batch4_gradacc4_decay0.05/checkpoint-1000
# RESULT_NAME=${lang}/n_${lang}_adaptgrpo_nomasft_lora_adaptsft1000mrm1000_5msteps_rlr1e-5_tdata400004langs_ga1_ro_lrconstant1e-5_1000steps_batch4_gradacc4_decay0.05_1000steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-4_tdata80004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_adaptgrpo_nomasft_lora_adaptsft100mrm200_5msteps_rlr1e-4_tdata80004langs_ga1_fr_lrconstant1e-5_100steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/multitask-adapt/bloom7b1_multitask-adaptdpo_tdata40000_lora_msft1mdpo400_2msteps_4langs_innerlr0_ga10_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05/checkpoint-2000
# RESULT_NAME=${lang}/n_${lang}_multitask-adaptdpo_tdata40000_lora_msft1mdpo400_2msteps_4langs_innerlr0_ga10_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05_2000steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata40000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05/checkpoint-2000
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata40000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05_2000steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/dpo_lora_sft1000_tdata40000all_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05/checkpoint-2000
# RESULT_NAME=${lang}/n_${lang}_dpo_lora_sft1000_tdata40000all_fr_lrconstant1e-5_2000steps_batch10_gradacc4_decay0.05_2000steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/fr/bloom7b1_adaptgrpo_lora_adaptsft100rm2004langs_ga1_fr_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/rerun_${lang}_adaptgrpo_lora_adaptsft100rm2004langs_ga1_fr_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/multitask-adapt/bloom7b1_multitask-adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr0_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/n_${lang}_multitask-adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr0_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/multitask-adapt/bloom7b1_multitask-adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr0_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/n_${lang}_multitask-adaptdpo_tdata100_lora_msft1mdpo400_2msteps_4langs_innerlr0_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/debug_bloom7b1_lora_sft100lr1e-4_sftrm200rlr1e-5batch40tdata8000_grpo_fr_lrconstant1e-4_100steps_batch4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/debug_${lang}_bloom7b1_lora_sft100lr1e-4_sftrm200rlr1e-5batch40tdata8000_grpo_fr_lrconstant1e-4_100steps_batch4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/debug_bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm500_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-4_200steps_batch4_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/debug_${lang}_bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm500_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-4_200steps_batch4_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16


# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/debug_bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm500_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-4_200steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}/debug_${lang}_bloom7b1_adaptgrpo_nomasft_lora_adaptsft100mrm500_5msteps_rlr1e-5_tdata80004langs_ga1_fr_lrconstant1e-4_200steps_batch4_gradacc4_decay0.05_100steps

# CUDA_VISIBLE_DEVICES=6 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo3_tdata100_lora_mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch16_gradacc1_decay0.05/checkpoint-200
RESULT_NAME=${lang}/${lang}_bloom7b1_adaptdpo3_tdata100_lora_mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch16_gradacc1_decay0.05_200steps

CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --policy_data_path=$policy_data_path \
        --batch_size=${BATCH_SIZE} \
        --prompt_length=${PROMPT_LEN} \
        --seq_length=${SEQ_LEN} \
        --eval_dataset_size=${EVAL_SIZE} \
        --result_name=${RESULT_NAME} \
        --bf16

policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo3_tdata100_lora_mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch16_gradacc1_decay0.05/checkpoint-100
RESULT_NAME=${lang}/${lang}_bloom7b1_adaptdpo3_tdata100_lora_mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch16_gradacc1_decay0.05_100steps

CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --policy_data_path=$policy_data_path \
        --batch_size=${BATCH_SIZE} \
        --prompt_length=${PROMPT_LEN} \
        --seq_length=${SEQ_LEN} \
        --eval_dataset_size=${EVAL_SIZE} \
        --result_name=${RESULT_NAME} \
        --bf16


policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo3_tdata8000_lora_mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_400steps_batch16_gradacc1_decay0.05/checkpoint-400
RESULT_NAME=${lang}/${lang}_bloom7b1_adaptdpo3_tdata8000_lora_mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_400steps_batch16_gradacc1_decay0.05_400steps

CUDA_VISIBLE_DEVICES=1 python eval_with_reward.py \
        --reward_path=$reward_path \
        --policy_path=$policy_path \
        --policy_data_path=$policy_data_path \
        --batch_size=${BATCH_SIZE} \
        --prompt_length=${PROMPT_LEN} \
        --seq_length=${SEQ_LEN} \
        --eval_dataset_size=${EVAL_SIZE} \
        --result_name=${RESULT_NAME} \
        --bf16

# lang=fr
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_dpo/bloom7b1_adaptdpo_tdata1000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05/checkpoint-200
# RESULT_NAME=${lang}/n_${lang}_adaptdpo_tdata1000_lora_msft1mdpo400_2msteps_4langs_innerlr3_ga10_fr_lrconstant1e-5_200steps_batch10_gradacc4_decay0.05_200steps

# CUDA_VISIBLE_DEVICES=4 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16


# lang=ca
# reward_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/reward_models/bloom1b7_judgerm_decay1e-6_ca_lr5e-5_10000steps_batch16_acc1/checkpoint-10000
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_adaptgrpo_nomasft_lora_adaptsft100rm2004langs_ga1_ca_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}_bloom7b1_adaptgrpo_nomasft_lora_adaptsft100rm2004langs_ga1_ca_lrconstant1e-4_100steps_batch4_gradacc4_decay0.05

# CUDA_VISIBLE_DEVICES=5 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16

# lang=ca
# reward_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/reward_models/bloom1b7_judgerm_decay1e-6_ca_lr5e-5_10000steps_batch16_acc1/checkpoint-10000
# policy_path=/users/staff/dmi-dmi/son0000/codes/MAML-Okapi/MAML_Okapi/ckpts/adaption_grpo/bloom7b1_lora_sft100lr1e-4_sftrm200batch40_grpo_ca_lrconstant1e-4_100steps_batch4_decay0.05/checkpoint-100
# RESULT_NAME=${lang}_bloom7b1_lora_sft100lr1e-4_sftrm200batch40_grpo_ca_lrconstant1e-4_100steps_batch4_decay0.05

# CUDA_VISIBLE_DEVICES=5 python eval_with_reward.py \
#         --reward_path=$reward_path \
#         --policy_path=$policy_path \
#         --policy_data_path=$policy_data_path \
#         --batch_size=${BATCH_SIZE} \
#         --prompt_length=${PROMPT_LEN} \
#         --seq_length=${SEQ_LEN} \
#         --eval_dataset_size=${EVAL_SIZE} \
#         --result_name=${RESULT_NAME} \
#         --bf16


