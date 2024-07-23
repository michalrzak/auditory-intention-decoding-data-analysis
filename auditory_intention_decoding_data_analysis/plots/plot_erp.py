# %% imports
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import mne
import numpy as np
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
    options: np.array
    targets: np.array

    def __post_init__(self) -> None:
        assert self.options.ndim == self.targets.ndim


# def plot_erp(average_option: np.array,
#              average_target: np.array,
#              channel_selection_indices: List[int],
#              channel_selection_labels: List[str],
#              output_file: Path) -> None:
#     def _verify_average(average: np.array) -> None:
#         assert average.ndim == 2
#         assert average.shape[0] == 126
#         assert average.shape[1] == 2501
#
#     _verify_average(average_option)
#     _verify_average(average_target)
#
#     average_option_selected = average_option[channel_selection_indices, :] * (10 ** 6)
#     average_target_selected = average_target[channel_selection_indices, :] * (10 ** 6)
#
#     fig, axs = plt.subplots(len(channel_selection_labels))
#     fig.set_figheight(9)
#     fig.set_figwidth(7.5)
#     for channel_option, channel_target, ax, label in zip(average_option_selected,
#                                                          average_target_selected,
#                                                          axs,
#                                                          channel_selection_labels):
#         ax.plot(channel_option, label="option")
#         ax.plot(channel_target, label="target")
#         ax.set_xticks(np.arange(0, 2500, 500), np.arange(-0.5, 2, 0.5))
#         ax.set_xlabel("t [s]")
#         ax.set_ylabel("U [μV]")
#         ax.set_title(label)
#
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.close()


def plot_erp(average_option: np.array,
             average_target: np.array,
             channel_selection_indices: List[int],
             channel_selection_labels: List[str],
             output_file: Path) -> None:
    def _verify_average(average: np.array) -> None:
        assert average.ndim == 2
        assert average.shape[0] == 126
        assert average.shape[1] == 2501

    _verify_average(average_option)
    _verify_average(average_target)

    plt.figure(figsize=(7.5, 3))
    plt.plot(np.mean(average_option, axis=0) * (10 ** 6), label="option")
    plt.plot(np.mean(average_target, axis=0) * (10 ** 6), label="target")
    plt.xticks(np.arange(0, 2500, 500), np.arange(-0.5, 2, 0.5))
    plt.xlabel("t [s]")
    plt.ylabel("U [μV]")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


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
        input_files: List[Path],
        output_dummy_file: Path
):
    assert all([input_file.is_file() for input_file in input_files])
    assert all([input_file.suffix == ".fif" for input_file in input_files])
    assert output_dummy_file.name == DUMMY_FILE_NAME

    output_folder = output_dummy_file.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    data_samples: List[Dict[str, Sample]] = []

    channel_labels = ["Cz", "CPz", "Pz"]
    channel_indices = None

    for input_file in input_files:
        raw = mne.io.read_raw_fif(input_file)
        if channel_indices is None:
            channel_indices = [raw.ch_names.index(ch) for ch in channel_labels]

        collected_erp = {}
        for name, triggers in TRIGGERS.items():
            events, event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX, baseline=None)

            options_epochs: mne.Epochs = epochs[f"Stimulus/S {triggers.option}"]
            targets_epochs: mne.Epochs = epochs[f"Stimulus/S {triggers.target}"]

            options = options_epochs.get_data()
            targets = targets_epochs.get_data()

            baseline_option = np.repeat(np.mean(options[:, :, :int((0 - TMIN) * 1000)], axis=2)[:, :, np.newaxis],
                                        options.shape[2], 2)
            baseline_target = np.repeat(np.mean(targets[:, :, :int((0 - TMIN) * 1000)], axis=2)[:, :, np.newaxis],
                                        targets.shape[2], 2)

            options_baselined = options - baseline_option
            targets_baselined = targets - baseline_target

            collected_erp[name] = Sample(options_baselined, targets_baselined)

        data_samples.append(collected_erp)

    assert channel_indices is not None
    assert len(channel_indices) == len(channel_labels)
    # %% collect across files

    samples_option: Dict[str, np.array] = defaultdict(lambda: np.zeros((126, int((TMAX - TMIN) * 1000 + 1))))
    n_samples_option: Dict[str, int] = defaultdict(lambda: 0)

    samples_target: Dict[str, np.array] = defaultdict(lambda: np.zeros((126, int((TMAX - TMIN) * 1000 + 1))))
    n_samples_target: Dict[str, int] = defaultdict(lambda: 0)

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

        plot_erp(o / n_o,
                 t / n_t,
                 channel_indices,
                 channel_labels,
                 output_folder / f"ERP-{tagger}.png")

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

    plot_erp(all_option / n_option,
             all_target / n_target,
             channel_indices,
             channel_labels,
             output_folder / "ERP-global.png")

    output_dummy_file.touch()


if __name__ == "__main__":
    app()
