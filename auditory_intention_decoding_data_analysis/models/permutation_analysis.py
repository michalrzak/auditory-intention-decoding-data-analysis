# %% Imports
import pickle
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn import metrics
from tqdm import tqdm

from auditory_intention_decoding_data_analysis.models.utils import Prediction

app = typer.Typer()


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


def get_f1_score(df: pd.DataFrame) -> float:
    return metrics.f1_score(df["labels"], df["predictions"], average="weighted")


def print_with_separator(df: pd.DataFrame) -> None:
    print(df)
    print("-------------------------")


def build_difference_map(accuracy_df: pd.DataFrame, taggers: Set[str]) -> Dict[Tuple[str, str], float]:
    return {ele: float(accuracy_df[ele[0]] - accuracy_df[ele[1]]) for ele in permutations(taggers, 2)}


def build_accuracy_df(prediction_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    counts_accuracy = None

    for cdf in prediction_dfs:
        ratio = get_accuracy(cdf)
        # print_with_separator(ratio)

        if counts_accuracy is None:
            counts_accuracy = pd.DataFrame(ratio)
        else:
            counts_accuracy = pd.concat([counts_accuracy, ratio], axis=1)

    assert counts_accuracy is not None
    return counts_accuracy


def build_f1_df(prediction_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    counts_f1 = None

    for cdf in prediction_dfs:
        f1_score_per_tagger = {tagger: get_f1_score(cdf[cdf["taggers"] == tagger]) for tagger in set(cdf["taggers"])}
        f1_score_df = pd.DataFrame.from_dict(f1_score_per_tagger, orient="index")
        # print_with_separator(f1_score_df)

        if counts_f1 is None:
            counts_f1 = f1_score_df
        else:
            counts_f1 = pd.concat([counts_f1, f1_score_df], axis=1)

    assert counts_f1 is not None
    return counts_f1


def shuffle_per_tagger(prediction_dfs: List[pd.DataFrame],
                       prediction_dfs_copy: List[pd.DataFrame],
                       rng: np.random.Generator) -> None:
    """The copy structure will get shuffled"""
    for cdf_copy, cdf in zip(prediction_dfs_copy, prediction_dfs):
        for tagger in set(cdf.taggers):
            cdf_copy.loc[cdf_copy.taggers == tagger, "predictions"] = (
                rng.permutation(cdf["predictions"][cdf["taggers"] == tagger]))


ACCURACY_ZERO_RULE_BASELINE = 2 / 3
ACCURACY_WEIGHTED_GUESSING_BASELINE = 5 / 9

F1_SCORE_ZERO_RULE_BASELINE = 8 / 15
F1_SCORE_WEIGHTED_GUESSING_BASELINE = 5 / 9


@app.command()
def main(prediction_path: Path):
    rng = np.random.default_rng(seed=42)

    with open(prediction_path, "rb") as file:
        predictions: List[Prediction] = pickle.load(file)

    prediction_dfs = [pd.DataFrame({
        "predictions": prediction.predictions,
        "labels":      prediction.labels,
        "taggers":     prediction.taggers
    }) for prediction in predictions]

    logger.info("Collecting accuracy counts")
    accuracy_df = build_accuracy_df(prediction_dfs)
    mean_accuracy_df = accuracy_df.mean(axis=1)

    logger.info("Printing accuracy count results")
    print_average_and_std(accuracy_df)

    logger.info("Computing permutation analysis")
    prediction_dfs_copy = deepcopy(prediction_dfs)

    n_permutations = 10000
    results_accuracy = {tagger: 0 for tagger in mean_accuracy_df.index}
    for i in tqdm(range(n_permutations)):
        shuffle_per_tagger(prediction_dfs, prediction_dfs_copy, rng)
        shuffled_accuracy_df = build_accuracy_df(prediction_dfs_copy)
        mean_shuffled_accuracy_df = shuffled_accuracy_df.mean(axis=1)

        results_accuracy = {t: results_accuracy[t] + int(mean_shuffled_accuracy_df[t] > mean_accuracy_df[t])
                            for t in results_accuracy.keys()}
    p_value_accuracy = {t: results_accuracy[t] / n_permutations for t in results_accuracy}
    print("=====  Accuracy p-values  =====")
    print(p_value_accuracy)
    print("===============================")

    logger.info("Collecting weighted f1-score counts")
    f1_df = build_f1_df(prediction_dfs)
    mean_f1_df = f1_df.mean(axis=1)

    logger.info("Printing weighted f1-score results")
    print_average_and_std(f1_df)

    logger.info("Computing permutation analysis")
    prediction_dfs_copy = deepcopy(prediction_dfs)

    n_permutations = 10000
    counts_f1 = {tagger: 0 for tagger in mean_f1_df.index}
    for i in tqdm(range(n_permutations)):
        shuffle_per_tagger(prediction_dfs, prediction_dfs_copy, rng)
        shuffled_f1_df = build_f1_df(prediction_dfs_copy)
        mean_shuffled_f1_df = shuffled_f1_df.mean(axis=1)

        counts_f1 = {t: counts_f1[t] + int(mean_shuffled_f1_df[t] > mean_f1_df[t])
                     for t in counts_f1.keys()}
    p_value_accuracy = {t: counts_f1[t] / n_permutations for t in counts_f1}
    print("=====  Accuracy p-values  =====")
    print(p_value_accuracy)
    print("===============================")

    logger.info("Doing Permutation analysis on first CV fold")
    n_permutations = 10000

    accuracy_baseline = get_accuracy(prediction_dfs[0])
    baseline_differences = build_difference_map(accuracy_baseline, set(prediction_dfs[0]["taggers"]))

    larger_count = {pair: 0 for pair in baseline_differences}

    for i in tqdm(range(n_permutations)):
        cdf = prediction_dfs[0].copy()

        cdf["taggers"] = rng.permutation(cdf["taggers"])
        accuracy = get_accuracy(cdf)

        differences = build_difference_map(accuracy, set(cdf["taggers"]))
        larger_count = {pair: larger_count[pair] + int(abs(differences[pair]) > abs(baseline_differences[pair]))
                        for pair in differences}

    larger_proportion = {pair: larger_count[pair] / n_permutations for pair in larger_count}

    logger.info("Printing permutation analysis results")

    print("\nCOUNTS:")
    print(larger_count)
    print("-------------------------")

    print("\nPROPORTIONS:")
    print(larger_proportion)
    print("-------------------------")

    # print("\n\n\n===================================")
    # print("The console now contains the following variables you to interact with:")
    # print("\tprediction_dfs: List[pd.DataFrame]; containing a list of dataframes containing the results of each CV")
    # print("\tlarger_count: Map[Tuple[str, str], int]; map containing how often the permuted tagger pair was larger")
    # print(f"\tlarger_proportion: Map[Tuple[str, str], float]; same as above but divided by {n_permutations}")
    # code.interact(local=locals())


if __name__ == '__main__':
    app()
