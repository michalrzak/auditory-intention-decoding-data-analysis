from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Prediction:
    predictions: np.ndarray
    labels: np.ndarray
    taggers: List[str]


def file_to_label_file(file: Path) -> Path:
    return file.parent / (str(file.stem) + "-labels.csv")


def read_labels_file(labels_file: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_file)
    assert "Target" in df.columns and "Tagger" in df.columns

    return df
