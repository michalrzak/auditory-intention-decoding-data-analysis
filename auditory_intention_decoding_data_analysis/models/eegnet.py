from pathlib import Path

import braindecode
import mne
import numpy as np
import torch
import typer
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file

mne.use_log_level("warning")
app = typer.Typer()

a = EEGNetv4(n_chans=126,
             n_outputs=2,
             n_times=3500,
             chs_info=None,  # TODO
             sfreq=1000)

cuda = torch.cuda.is_available()
device = "cpu"  # "cuda" if cuda else "cpu"

# Set random seed to be able to reproduce results
seed = 42

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameter
LEARNING_RATE = 0.0625 * 0.01  # parameter taken from Braindecode
WEIGHT_DECAY = 0  # parameter taken from Braindecode
BATCH_SIZE = 200
EPOCH = 10  # 1000
PATIENCE = 3  # 300
fmin = 4
fmax = 100
tmin = 0
tmax = None


def read_labels_file(labels_file: Path) -> list[int]:
    with open(labels_file, "r") as file:
        headers = file.readline()
        assert headers.strip() == "Label"

        return [int(line.strip()) for line in file.readlines()]


@app.command()
def main(
        input_files: list[Path],
        output_file: Path,
):
    labels_files = [file_to_label_file(input_file) for input_file in input_files]

    assert all(input_file.name.endswith("-epo.fif") for input_file in input_files)
    assert all(labels_file.exists() for labels_file in labels_files)
    # assert output_file is something as well

    epochs = [mne.read_epochs(input_file) for input_file in input_files]
    epochs_combined: mne.Epochs = mne.concatenate_epochs(epochs)
    dataset = braindecode.datasets.create_from_mne_epochs(epochs, 3501, 8, False)
    labels = sum([read_labels_file(file) for file in labels_files], [])

    clf = EEGClassifier(
            module=EEGNetv4(n_chans=len(epochs_combined.ch_names),
                            n_outputs=2,
                            n_times=len(epochs_combined.times),
                            chs_info=epochs_combined.info["chs"],
                            input_window_seconds=epochs_combined.tmax - epochs_combined.tmin + 0.001,
                            sfreq=epochs_combined.info["sfreq"]),
            optimizer=torch.optim.Adam,
            optimizer__lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_epochs=EPOCH,
            train_split=ValidSplit(0.2, random_state=seed),
            device=device,
            compile=True,
            callbacks=[
                EarlyStopping(monitor="valid_loss", patience=PATIENCE),
                EpochScoring(
                        scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
                ),
                EpochScoring(
                        scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
                ),
            ],
            verbose=1,
    )

    clf.fit(dataset, np.array(labels))


if __name__ == '__main__':
    app()
