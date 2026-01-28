<h1 align="center"> <p> Meta-Learning Preferences for Multilingual LLM Alignment  </p></h1>
<h3 align="center">
    <p>Data-Efficient Preference Learning for Multilingual LLMs through MAML-based pipelines</p>
</h3>

The base implementation the code is taken from [Okapi paper (Dac et al., 2023)](https://github.com/nlp-uoregon/Okapi).
We use [mlmm-evaluation repo](https://github.com/nlp-uoregon/mlmm-evaluation) for evaluating our models on multilingual OOD benchmarks.

# Preparation
## Download Datasets
The entire dataset includes
- **multilingual-alpaca-52k** for supervised fine-tuning
- **multilingual-ranking-data-42k** for reward modelling
- **multilingual-rl-tuning-64k** for rl training
```bash
bash scripts/download.sh
```

## Set up the environment
```bash
pip install -r requirements.txt
```

## Train a judge reward model for each target language
```bash
# Make sure you configure the content of the script to change the target language.
bash scripts/bloom_full_reward10k.sh
```

# Training
We provide sample scripts for running Romanian experiments with Gemma3-270M, and Catalan experiments with BLOOM-7B1. 
Evaluation results are automatically computed at the end of each run and saved to `./eval_results/{lang}/summary/summary.csv`.

## Gemma3-270m Policy Experiments
We provide sample scripts for the Romanian task.
1. baseline DPO
```bash
bash scripts/baseline_dpo_gemma_ro.sh
```

2. Baseline RLHF
```bash
bash scripts/baseline_rlhf_gemma_ro.sh
```

3. Meta-DPO
```bash
bash scripts/meta_dpo_gemma_ro.sh
```

4. Meta-RLHF
```bash
bash scripts/meta_rlhf_gemma_ro.sh
```

## Bloom-7b1 Policy Experiments
We provide sample scripts for the Catalan task.
1. baseline DPO
```bash
bash scripts/ca_bloom7b1_baselinedpo.sh
```

2. Baseline RLHF
```bash
bash scripts/ca_bloom7b1_baselinerlhf.sh
```

3. Meta-DPO
```bash
bash scripts/ca_bloom7b1_mamldpo_lora_pipeline.sh
```

4. Meta-RLHF
```bash
bash scripts/ca_bloom7b1_mamlrlhf_lora_entire_pipeline.sh
```