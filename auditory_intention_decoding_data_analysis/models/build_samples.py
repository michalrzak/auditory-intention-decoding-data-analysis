import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import typer
from loguru import logger

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file

mne.use_log_level("warning")
app = typer.Typer()


@dataclass
class Tagger:
    option: int
    target: int


TRIGGERS = {
    "RAW":      Tagger(option=50, target=70),
    "AM":       Tagger(option=51, target=71),
    "FM":       Tagger(option=52, target=72),
    "BINAURAL": Tagger(option=56, target=76)
}


@app.command()
def main(
        input_file: Path,
        output_file: Path,
        tmin: float,
        tmax: float,
        sfreq: Optional[int] = None
):
    # tmin = float(tmin)
    # tmax = float(tmax)

    assert input_file.is_file()
    assert input_file.suffix == ".fif"
    assert output_file.name.endswith("-epo.fif")

    logger.info(f"Started building features (extracting mne Epochs from: {input_file}")
    raw: mne.io.Raw = mne.io.read_raw_fif(input_file)
    if sfreq is not None:
        raw.load_data()
        raw.resample(sfreq)

    logger.info("Extracting events")
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=(None, 0), event_repeated="drop")

    epochs_options: mne.Epochs = epochs[[f"Stimulus/S {triggers.option}" for triggers in TRIGGERS.values()]]
    epochs_targets: mne.Epochs = epochs[[f"Stimulus/S {triggers.target}" for triggers in TRIGGERS.values()]]

    epochs_combined = mne.epochs.concatenate_epochs([epochs_options, epochs_targets])
    labels = np.array([0] * len(epochs_options) + [1] * len(epochs_targets))

    logger.info("Saving epochs")
    epochs_combined.save(output_file, overwrite=True)
    with open(file_to_label_file(output_file), "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["Target", "Tagger"])
        writer.writerows([[int(ele), ] for ele in labels])


if __name__ == '__main__':
    app()
