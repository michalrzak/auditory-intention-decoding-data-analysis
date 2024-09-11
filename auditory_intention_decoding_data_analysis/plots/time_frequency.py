import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import typer
from loguru import logger
from toolz import pipe

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file, read_labels_file

app = typer.Typer()


def do_analysis(windows: np.array,
                fs: int,
                step_s: float,
                window_s: float,
                max_freq: float,
                freq_resolution: float):
    window_step = int(step_s * fs)
    samples_per_window = int(window_s * fs)

    results_channels = windows.shape[1]
    result_times = int(np.ceil((windows.shape[2] - samples_per_window) / window_step))
    result_freqs = int(max_freq / freq_resolution)
    result_matrix = np.empty((results_channels, result_times, result_freqs))

    for (i, sample_start) in enumerate(range(0, windows.shape[2] - samples_per_window, window_step)):
        sub_window = windows[:, :, sample_start:sample_start + samples_per_window]
        power_spectrum = windows_to_spectrum(sub_window, fs, max_freq, freq_resolution)
        assert power_spectrum.shape == (windows.shape[1], result_freqs)

        result_matrix[:, i, :] = power_spectrum

    return result_matrix


def windows_to_spectrum(windows: np.array,
                        fs: int,
                        max_freq: float,
                        freq_resolution: float):
    assert windows.ndim == 3

    sample_spectrum = pipe(windows,
                           lambda x: np.fft.fft(x, axis=2),
                           lambda x: np.fft.fftshift(x, axes=2))

    f_resolution = fs / windows.shape[2]

    start = int(sample_spectrum.shape[2] / 2)
    stop = int(start + max_freq / f_resolution)
    step = int(freq_resolution / f_resolution)

    current_spectrum = pipe(sample_spectrum,
                            lambda x: x[:, :, start:stop],
                            lambda x: np.pad(x,
                                             [(0, 0), (0, 0), (0, step - ((stop - start) % step))],
                                             mode="constant"),
                            lambda x: x.reshape((x.shape[0], x.shape[1], -1, step)),
                            lambda x: np.sum(x, axis=3),
                            lambda x: x ** 2,
                            np.abs,
                            np.log)

    return np.average(current_spectrum, axis=0)[:, :int(max_freq / freq_resolution)]


@app.command()
def main(epoch_files: List[pathlib.Path],
         output_folder: pathlib.Path) -> None:
    logger.info("Started computing time frequency!")
    labels_files = [file_to_label_file(epoch_file) for epoch_file in epoch_files]

    assert all(epoch_file.name.endswith("-epo.fif") for epoch_file in epoch_files)
    assert all(labels_file.exists() for labels_file in labels_files)

    labels_list = [read_labels_file(file) for file in labels_files]

    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Loading epochs")
    epochs_list = [mne.read_epochs(epoch_file) for epoch_file in epoch_files]

    options: Optional[np.array] = None
    targets: Optional[np.array] = None
    channel_names: Optional[List[str]] = None
    f_sampling: Optional[int] = None
    for epochs, labels_df in zip(epochs_list, labels_list):
        if f_sampling is None:
            f_sampling = int(epochs.info["sfreq"])
        if channel_names is None:
            channel_names = epochs.info['ch_names']

        labels = labels_df["Target"]

        all_data = epochs.get_data()
        assert all_data.shape[0] == labels.shape[0]

        temp_options = all_data[labels == 0, :, :]
        temp_targets = all_data[labels == 1, :, :]

        if options is None:
            assert targets is None

            empty_shape = (0, temp_options.shape[1], temp_options.shape[2])
            options = np.empty(empty_shape)
            targets = np.empty(empty_shape)

        options = np.append(options, temp_options, axis=0)
        targets = np.append(targets, temp_targets, axis=0)

    f_sampling = 1000

    options_tfr = do_analysis(options, f_sampling, 0.05, 2, 50, 1)
    targets_tfr = do_analysis(targets, f_sampling, 0.05, 2, 50, 1)

    difference_tfr = targets_tfr - options_tfr

    mean_difference_tfr = np.mean(difference_tfr[:, :, :], axis=0)
    scale = np.max(np.abs(mean_difference_tfr))

    plt.figure(figsize=(9, 7))
    plt.title("Grand average")
    plt.imshow(mean_difference_tfr.T, aspect="auto", cmap="coolwarm", origin="lower", vmin=-scale, vmax=scale)
    plt.xticks(np.arange(0, mean_difference_tfr.shape[0], mean_difference_tfr.shape[0] / 10),
               [f"{ele:.2f}" for ele in np.arange(-1.5, 3.01, 0.5)])
    plt.colorbar()
    plt.savefig(output_folder / "grand_average.png")
    plt.close()

    for i, channel in enumerate(channel_names):
        plt.figure(figsize=(9, 7))
        plt.title(f"{channel}")
        plt.imshow(difference_tfr[i, :, :].T, aspect="auto", cmap="coolwarm", origin="lower", vmin=-scale, vmax=scale)
        plt.xticks(np.arange(0, difference_tfr[i, :, :].shape[0], difference_tfr[i, :, :].shape[0] / 10),
                   [f"{ele:.2f}" for ele in np.arange(-1.5, 3.01, 0.5)])
        plt.colorbar()
        plt.savefig(output_folder / f"{channel}.png")
        plt.close()


if __name__ == '__main__':
    app()
