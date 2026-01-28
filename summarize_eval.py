import numpy as np
import pandas as pd
from glob import glob
import argparse
import os

# path_dfs = "./eval_results/"
# path_res = path_dfs + "summary/summary.csv"
# # path_ref = path_dfs + "fr_judgerm_apertus8bit_100prompts.csv"
# # path_ref = path_dfs + "fr_judgerm_bloom7b1_100prompts.csv"
# # path_ref = path_dfs + "fr_judgerm_bloom7b1-base_1000prompts.csv"
# path_ref = path_dfs + "ca_judgerm_bloom7b1-base_1000prompts.csv"

def get_paths(lang):
    path_dfs = f"./eval_results/{lang}/"
    path_res = path_dfs + "summary/summary.csv"
    path_ref = path_dfs + f"{lang}_judgerm_bloom7b1-base_1000prompts.csv"
    return path_dfs, path_res, path_ref

# path_ref = path_dfs + "fr_reward10k_base_100prompts.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="")
    args = parser.parse_args()
    path_dfs, path_res, path_ref = get_paths(args.lang)
    l_dfs = glob(path_dfs + "*.csv")

    df = {
        "name_file": list(),
        "reward_model": list(),
        "policy_model": list(),
        "winrate_mean": list(),
        "reward_mean": list(),
        "winrate_stderror": list(),
        "reward_std": list(),
        "reward_stderror": list(),
        
    }

    df_ref = pd.read_csv(path_ref)

    for path in l_dfs:
        try:
            df_i = pd.read_csv(path)
            rewards_mean = df_i["reward"].mean()
            rewards_std = df_i["reward"].std()
            rewards_stderror = rewards_std / np.sqrt(df_i.shape[0])
            reward_diff = df_i["reward"] - df_ref["reward"]
            wins = (reward_diff > 0) * 1.0 + (reward_diff == 0) * 0.5
            winrate_mean = wins.mean()
            winrate_stderror = wins.std() / np.sqrt(df_i.shape[0])

            df["name_file"].append(path.split("/")[-1])
            df["policy_model"].append(df_i["policy_model"][0])
            df["reward_model"].append(df_i["reward_model"][0])
            df["reward_mean"].append(rewards_mean)
            df["reward_std"].append(rewards_std)
            df["reward_stderror"].append(rewards_stderror)
            df["winrate_mean"].append(winrate_mean)
            df["winrate_stderror"].append(winrate_stderror)

            print(f"Complete: {path}")

        except Exception as e:
            print(path)
            print(e)

    df = pd.DataFrame(df).sort_values(by=["policy_model"])
    os.makedirs(path_dfs + "summary/", exist_ok=True)
    df.to_csv(path_res, index=False)