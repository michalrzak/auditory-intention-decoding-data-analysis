# %% imports
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import numpy.typing as npt
import typer

from auditory_intention_decoding_data_analysis.config import DUMMY_FILE_NAME

mne.use_log_level("warning")
app = typer.Typer()


@dataclass
class Tagger:
    option: int
    target: int


@dataclass
class Sample:
    options: npt.NDArray
    targets: npt.NDArray

    def __post_init__(self) -> None:
        assert self.options.ndim == self.targets.ndim


TRIGGERS = {
    "RAW":      Tagger(option=50, target=70),
    "AM":       Tagger(option=51, target=71),
    "FM":       Tagger(option=52, target=72),
    "BINAURAL": Tagger(option=56, target=76)
}

TMIN = -0.5
TMAX = 2

F_SAMPLING = 1000


@app.command()
def main(
        input_files: list[Path],
        output_dummy_file: Path
):
    assert all([input_file.is_file() for input_file in input_files])
    assert all([input_file.suffix == ".fif" for input_file in input_files])
    assert output_dummy_file.name == DUMMY_FILE_NAME

    output_folder = output_dummy_file.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    data_samples: list[dict[str, Sample]] = []

    for input_file in input_files:
        raw = mne.io.read_raw_fif(input_file)

        collected_erp = {}
        for name, triggers in TRIGGERS.items():
            events, event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX, baseline=None)

            options_epochs: mne.Epochs = epochs[f"Stimulus/S {triggers.option}"]
            targets_epochs: mne.Epochs = epochs[f"Stimulus/S {triggers.target}"]

            options = options_epochs.get_data(copy=False)
            targets = targets_epochs.get_data(copy=False)

            baseline_option = np.repeat(np.mean(options[:, :, :int((0 - TMIN) * 1000)], axis=2)[:, :, np.newaxis],
                                        options.shape[2], 2)
            baseline_target = np.repeat(np.mean(targets[:, :, :int((0 - TMIN) * 1000)], axis=2)[:, :, np.newaxis],
                                        targets.shape[2], 2)

            options_baselined = options - baseline_option
            targets_baselined = targets - baseline_target

            collected_erp[name] = Sample(options_baselined, targets_baselined)

        data_samples.append(collected_erp)

    # %% collect across files

    samples_option: dict[str, npt.NDArray] = defaultdict(lambda: np.zeros((126, int((TMAX - TMIN) * 1000 + 1))))
    n_samples_option: dict[str, int] = defaultdict(lambda: 0)

    samples_target: dict[str, npt.NDArray] = defaultdict(lambda: np.zeros((126, int((TMAX - TMIN) * 1000 + 1))))
    n_samples_target: dict[str, int] = defaultdict(lambda: 0)

    for data_sample in data_samples:

        for name, sample in data_sample.items():
            samples_option[name] += np.sum(sample.options, axis=0)
            n_samples_option[name] += sample.options.shape[0]

            samples_target[name] += np.sum(sample.targets, axis=0)
            n_samples_target[name] += sample.targets.shape[0]

    # %% plot per tagger
    for tagger in samples_option:
        o = samples_option[tagger]
        t = samples_target[tagger]

        n_o = n_samples_option[tagger]
        n_t = n_samples_target[tagger]

        plt.plot(np.mean(o / n_o, axis=0))
        plt.plot(np.mean(t / n_t, axis=0))
        plt.savefig(output_folder / f"ERP-{tagger}.png")
        plt.close()

    # %% collect across taggers

    all_option = np.zeros((126, int((TMAX - TMIN) * 1000 + 1)))
    n_option = 0
    all_target = np.zeros((126, int((TMAX - TMIN) * 1000 + 1)))
    n_target = 0

    for sample, n in zip(samples_option.values(), n_samples_option.values()):
        all_option += sample
        n_option += n

    for sample, n in zip(samples_target.values(), n_samples_target.values()):
        all_target += sample
        n_target += n

    # %% do plots
    plt.plot(np.mean(all_option / n_option, axis=0))
    plt.plot(np.mean(all_target / n_target, axis=0))
    plt.savefig(output_folder / "ERP-global.png")
    plt.close()

    output_dummy_file.touch()


if __name__ == "__main__":
    app()
