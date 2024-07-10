# %% Imports
import pickle
from itertools import permutations
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics

from auditory_intention_decoding_data_analysis.models.eegnet2 import Prediction


# %% Define helper functions
def print_average_and_std(df: pd.DataFrame) -> None:
    print("==========  AVERAGE  ==========")
    print(df.mean(axis=1))
    print("===============================")
    print("============  STD  ============")
    print(df.std(axis=1))
    print("===============================")


def get_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    correct_counts = df[df["predictions"] == df["labels"]]["taggers"].value_counts()
    all_counts = df["taggers"].value_counts()
    return correct_counts / all_counts


def print_with_separator(df: pd.DataFrame) -> None:
    print(df)
    print("-------------------------")


def build_difference_map(accuracy_df: pd.DataFrame, taggers: Set[str]) -> Dict[Tuple[str, str], float]:
    return {ele: float(accuracy_df[ele[0]] - accuracy_df[ele[1]]) for ele in permutations(taggers, 2)}


# %% Load
with open("eegnet_results/prediction.pkl", "rb") as file:
    predictions: List[Prediction] = pickle.load(file)

prediction_dfs = [pd.DataFrame({
    "predictions": prediction.predictions,
    "labels":      prediction.labels,
    "taggers":     prediction.taggers
}) for prediction in predictions]

# %% Print counts accuracy
counts_accuracy = None

for cdf in prediction_dfs:
    ratio = get_accuracy(cdf)
    print_with_separator(ratio)

    counts_accuracy = pd.DataFrame(ratio) if counts_accuracy is None else pd.concat([counts_accuracy, ratio], axis=1)

print_average_and_std(counts_accuracy)


# %% Print counts f1
def get_f1_score(df: pd.DataFrame) -> float:
    return metrics.f1_score(df["labels"], df["predictions"], average="weighted")


counts_f1 = None

for cdf in prediction_dfs:
    f1_score_per_tagger = {tagger: get_f1_score(cdf[cdf["taggers"] == tagger]) for tagger in set(cdf["taggers"])}
    f1_score_df = pd.DataFrame.from_dict(f1_score_per_tagger, orient="index")
    print_with_separator(f1_score_df)

    counts_f1 = f1_score_df if counts_f1 is None else pd.concat([counts_f1, f1_score_df], axis=1)

print_average_and_std(counts_f1)

# %% Permutation fun
n_permutations = 10000

rng = np.random.default_rng(seed=42)
accuracy_baseline = get_accuracy(prediction_dfs[0])
baseline_differences = build_difference_map(accuracy_baseline, set(prediction_dfs[0]["taggers"]))

larger_count = {pair: 0 for pair in baseline_differences}

for i in range(n_permutations):
    cdf = prediction_dfs[0].copy()

    cdf["taggers"] = np.random.permutation(cdf["taggers"])
    accuracy = get_accuracy(cdf)
    # print_with_separator(accuracy)

    differences = build_difference_map(accuracy, set(cdf["taggers"]))
    larger_count = {pair: larger_count[pair] + int(baseline_differences[pair] > differences[pair])
                    for pair in differences}
