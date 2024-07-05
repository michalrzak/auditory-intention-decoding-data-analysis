from pathlib import Path
from typing import List

import mne
import numpy as np
import typer
from tensorflow.keras import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from auditory_intention_decoding_data_analysis.external.EEGModels import EEGNet
from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file

mne.use_log_level("warning")
app = typer.Typer()


def read_labels_file(labels_file: Path) -> List[int]:
    with open(labels_file, "r") as file:
        headers = file.readline()
        assert headers.strip() == "Label"

        return [int(line.strip()) for line in file.readlines()]


@app.command()
def main(
        input_files: List[Path],
        output_file: Path,
):
    labels_files = [file_to_label_file(input_file) for input_file in input_files]

    assert all(input_file.name.endswith("-epo.fif") for input_file in input_files)
    assert all(labels_file.exists() for labels_file in labels_files)

    epochs = [mne.read_epochs(input_file) for input_file in input_files]
    assert all(epoch.info["sfreq"] == 128 for epoch in epochs)

    epochs_combined: mne.Epochs = mne.concatenate_epochs(epochs)
    labels_combined = np.array(sum([read_labels_file(file) for file in labels_files], []))

    data = epochs_combined.get_data()
    data_shaped = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

    rng = np.random.default_rng()
    permutations = rng.permutation(data_shaped.shape[0])

    data_shuffled = data_shaped[permutations, :, :, :]
    labels_shuffled = labels_combined[permutations]

    n_data_train = int(data_shuffled.shape[0] * 0.8)

    data_train = data_shuffled[:n_data_train, :, :, :]
    labels_train = labels_shuffled[:n_data_train]

    data_test = data_shuffled[n_data_train:, :, :, :]
    labels_test = data_shuffled[n_data_train:]

    model: Model = EEGNet(2, 126, int(3.5 * 128 + 1))  # TODO: parameters

    model.compile(Adam(), loss="mse")
    model.fit(data_train, labels_train)
    model.save("test.keras")


if __name__ == '__main__':
    app()
