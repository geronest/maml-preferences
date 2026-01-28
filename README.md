<h1 align="center"> <p> Meta-Learning Preferences for Multilingual LLMs  </p></h1>
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

1. Supervised Fine-tuning
```bash
bash scripts/supervised_finetuning.sh
```

2. Reward modelling
```bash
bash scripts/reward_modeling.sh
```

1. Fine-tuning with RLHF
```bash
bash scripts/rl_training.sh
```

For Bloom-7.1B model experiments, sample scripts are provided in `scripts/` with names starting with `ca_*`. 