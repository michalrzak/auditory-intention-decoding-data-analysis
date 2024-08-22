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


def windows_to_tfr(windows: np.array,
                   fs,
                   step_s: float,
                   max_freq: float,
                   freq_resolution: float):
    assert windows.ndim == 3

    n_times = int(windows.shape[2] / step_s)

    sample_spectrum = pipe(windows,
                           lambda x: np.fft.fft(x, axis=2),
                           lambda x: np.fft.fftshift(x, axes=2))

    f_resolution = fs / windows.shape[2]

    start = int(sample_spectrum.shape[2] / 2)
    stop = int(start + max_freq / f_resolution)
    step = int(freq_resolution / f_resolution)

    current_time_freq = pipe(sample_spectrum,
                             lambda x: x[:, :, start:stop],
                             lambda x: np.pad(x,
                                              [(0, 0), (0, 0), (0, step - ((stop - start) % step))],
                                              mode="constant"),
                             lambda x: x.reshape((x.shape[0], x.shape[1], -1, step)),
                             lambda x: np.sum(x, axis=3),
                             lambda x: x ** 2,
                             np.abs,
                             np.log)

    return current_time_freq


# def collect_option_and_target(
#         samples: Iterable[Sample],
#         start_s: float,
#         stop_s: float,
#         step_s: float,
#         max_freq: float,
#         freq_resolution: float) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
#     n_times = round((stop_s - start_s) / step_s)
#     n_freqs = round(max_freq / freq_resolution)
#
#     option = np.zeros((n_freqs, n_times, 0))
#     target = np.zeros((n_freqs, n_times, 0))
#     window_length = 2
#
#     for sample in samples:
#         for k, option_interval in enumerate(sample.option_intervals):
#
#             sample_time_freq = np.zeros((n_freqs, n_times))
#
#             for (i, time) in enumerate(np.arange(start_s, stop_s, step_s)):
#                 sample_min = int(option_interval[0] + time * data.fs)
#                 sample_max = int(sample_min + window_length * data.fs)
#
#                 sample_spectrum = pipe(data.array,
#                                        lambda x: x[:, sample_min:sample_max],
#                                        lambda x: np.fft.fft(x, axis=1),
#                                        np.fft.fftshift)
#
#                 f_resolution = data.fs / (sample_max - sample_min)
#
#                 start = int(sample_spectrum.shape[1] // 2)
#                 stop = int(start + max_freq / f_resolution)
#                 step = int(freq_resolution / f_resolution)
#
#                 current_time_freq = pipe(sample_spectrum,
#                                          lambda x: x[:, start:stop:step] + x[:, start + 1:stop:step],
#                                          lambda x: x ** 2,
#                                          np.abs,
#                                          np.log,
#                                          lambda x: np.average(x, axis=0))
#
#                 sample_time_freq[i, :] += current_time_freq
#
#             sample_time_freq_reshaped = np.reshape(sample_time_freq, (n_freqs, n_times, 1))
#             if k == sample.i_target:
#                 target = np.append(target, sample_time_freq_reshaped, axis=2)
#             else:
#                 option = np.append(option, sample_time_freq_reshaped, axis=2)
#
#     return option, target
#

#
# def do_analysis(samples: Iterable[Sample],
#                 start_s: float,
#                 stop_s: float,
#                 step_s: float,
#                 max_freq: float,
#                 freq_resolution: float,
#                 plot_title: str,
#                 export_file: pathlib.Path) -> None:
#     option, target = collect_option_and_target(samples,
#                                                start_s,
#                                                stop_s,
#                                                step_s,
#                                                max_freq,
#                                                freq_resolution)
#
#     std_option = np.std(option, axis=2)
#     std_target = np.std(target, axis=2)
#
#     log_psd_diff = (np.average(target, axis=2) - np.average(option, axis=2)).T
#     log_psd_diff = np.divide(log_psd_diff, std_option)
#
#     fig, ax = plt.subplots(1, 3)
#     fig.set_figwidth(30)
#     fig.set_figheight(7)
#     fig.suptitle(plot_title)
#
#     plot_time_frequency(log_psd_diff, start_s, stop_s, ax[0], fig)
#     ax[0].set_title("Log-PSD target - option")
#
#     plot_time_frequency(std_option, start_s, stop_s, ax[1], fig)
#     ax[1].set_title("STD options")
#
#     plot_time_frequency(std_target, start_s, stop_s, ax[2], fig)
#     ax[2].set_title("STD target")
#
#     export_file.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(export_file)
#     plt.close()
#

@app.command()
def main(epoch_files: List[pathlib.Path],
         output_folder: pathlib.Path,
         max_freq: int) -> None:
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

    f_sampling = 128

    options_tfr = windows_to_tfr(options, f_sampling, 1, 50, 1)
    targets_tfr = windows_to_tfr(targets, f_sampling, 1, 50, 1)

    avg_options_tfr = np.average(options_tfr, axis=0)
    avg_targets_tfr = np.average(targets_tfr, axis=0)

    avg_difference_tfr = avg_targets_tfr - avg_options_tfr

    options_tfr = mne.time_frequency.tfr_array_multitaper(options, f_sampling, np.arange(2, max_freq),
                                                          output='avg_power_itc')
    targets_tfr = mne.time_frequency.tfr_array_multitaper(targets, f_sampling, np.arange(2, max_freq),
                                                          output='avg_power_itc')

    # the output of the function above is complex number, where the real component corresponds to the TFR and the
    #  imaginary to the ITC
    average_difference = np.real(targets_tfr) - np.real(options_tfr)

    current_tfr = np.average(average_difference[[11, 12], :, :], axis=0)
    scale = np.max(np.abs(current_tfr))

    plt.figure(figsize=(20, 10))
    # plt.title(channel)
    plt.imshow(current_tfr, aspect="auto", cmap="coolwarm", origin="lower", vmax=scale, vmin=-scale)
    plt.xticks(np.arange(0, 4000, 200), [f"{ele:.2f}" for ele in np.arange(-1, 3, 0.2)])
    plt.colorbar()
    plt.savefig(output_folder / "all.png")
    plt.close()

    for i, channel in enumerate(channel_names):
        current_tfr = average_difference[i, :, :]
        scale = np.max(np.abs(current_tfr))

        plt.figure(figsize=(20, 10))
        plt.title(channel)
        plt.imshow(current_tfr, aspect="auto", cmap="coolwarm", origin="lower", vmax=scale, vmin=-scale)
        plt.xticks(np.arange(0, 4000, 200), [f"{ele:.2f}" for ele in np.arange(-1, 3, 0.2)])
        plt.colorbar()
        plt.savefig(output_folder / f"{channel}.png")
        plt.close()


if __name__ == '__main__':
    app()
