import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mne
import numpy as np
import typer
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd

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
        output_folder: Path,
):
    labels_files = [file_to_label_file(input_file) for input_file in input_files]

    assert all(input_file.name.endswith("-epo.fif") for input_file in input_files)
    assert all(labels_file.exists() for labels_file in labels_files)

    epochs = [mne.read_epochs(input_file) for input_file in input_files]
    assert all(epoch.info["sfreq"] == 128 for epoch in epochs)

    epochs_combined: mne.Epochs = mne.concatenate_epochs(epochs)
    labels_combined = np.array(sum([read_labels_file(file) for file in labels_files], []))
    labels_onehot = np.stack([-labels_combined + 1, labels_combined], axis=1)

    data = epochs_combined.get_data()

    scaler = mne.decoding.Scaler(scalings="median")
    data_scaled = scaler.fit_transform(data)

    data_shaped = data_scaled.reshape((data_scaled.shape[0], data_scaled.shape[1], data_scaled.shape[2], 1))

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=45)
    sss.get_n_splits(data_shaped, labels_onehot)

    splits = sss.split(data_shaped, labels_onehot)

    histories = []
    accuracies = []
    f1_scores = []
    for i, split in enumerate(splits):

        data_train = data_shaped[split[0]]
        labels_train = labels_onehot[split[0]]

        data_test = data_shaped[split[1]]
        labels_test = labels_onehot[split[1]]

        model: Model = EEGNet(2, 126, int(3.5 * 128 + 1))  # TODO: parameters

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy']) # binary_crossentropy

        history = model.fit(data_train, labels_train, epochs=500, batch_size=32, verbose=1)
        histories.append(history.history)

        argmax_labels = np.argmax(labels_test, axis=1)
        argmax_predictions = np.argmax(model.predict(data_test), axis=1)
        accuracies.append(metrics.accuracy_score(argmax_labels, argmax_predictions))
        f1_scores.append(metrics.f1_score(argmax_labels, argmax_predictions, average='weighted'))

        model.save(output_folder / f"model-split-{i}.keras")

    with open(output_folder / "history.pkl", "wb") as file:
        pickle.dump(histories, file)

    with open(output_folder / "accuracies.pkl", "wb") as file:
        pickle.dump(accuracies, file)

    with open(output_folder / "f1.pkl", "wb") as file:
        pickle.dump(f1_scores, file)

    history_accuracies = pd.DataFrame([history["accuracy"] for history in histories]).melt()
    history_loss = pd.DataFrame([history["loss"] for history in histories]).melt()

    sns.lineplot(history_accuracies, x="variable", y="value", label="accuracy")
    sns.lineplot(history_loss, x="variable", y="value", label="loss")
    plt.legend()
    plt.savefig(output_folder / "training_plot.png")
    plt.close()

    pass

if __name__ == '__main__':
    app()
