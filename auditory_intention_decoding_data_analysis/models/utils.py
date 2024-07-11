from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Prediction:
    predictions: np.ndarray
    labels: np.ndarray
    taggers: List[str]


def file_to_label_file(file: Path) -> Path:
    return file.parent / (str(file.stem) + "-labels.csv")
