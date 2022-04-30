from py_utils import mso_utils
from py_utils.mso_utils import *

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

mso_settings = {
    "sr": 44100,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 512,
    "n_bins_per_octave": 16,
    "n_octaves": 9,
    "f_min": 40,
    "mean_filter_size": 22,
    "c_freq": [55, 90, 138, 175, 350, 6000, 8500, 12500]
}

def mso(y, grid_lines, **kwargs):
    """
    Multiband synthesized onsets.
    """
    sr = kwargs.get('sr', 44100)
    n_fft = kwargs.get('n_fft', 1024)
    win_length = kwargs.get('win_length', 1024)
    hop_length = kwargs.get('hop_length', 512)
    n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
    n_octaves = kwargs.get('n_octaves', 9)
    f_min = kwargs.get('f_min', 40)
    mean_filter_size = kwargs.get('mean_filter_size', 22)
    c_freq = kwargs.get('c_freq', [55, 90, 138, 175, 350, 6000, 8500, 12500])

    # if the audio starts right at grid-line 0, but the grid lines are relative to -0.5 microtiming of first grid line, set to True
    reorder_to_start_before_gridline_0 = kwargs.get('reorder_to_start_before_gridline_0', True)
    if reorder_to_start_before_gridline_0 is True:
        half_grid_res_in_samples = int((grid_lines[1]-grid_lines[0]) * sr / 2.0)
        y = np.roll(y, half_grid_res_in_samples)    # grab last 32note segment and put at beginning

    # onset strength spectrogram
    spec, f_cq = get_onset_strength_spec(y, n_fft=n_fft, win_length=win_length,
                                         hop_length=hop_length, n_bins_per_octave=n_bins_per_octave,
                                         n_octaves=n_octaves, f_min=f_min, sr=sr,
                                         mean_filter_size=mean_filter_size)

    # multiband onset detection and strength
    mb_onset_strength = reduce_f_bands_in_spec(c_freq, f_cq, spec)
    mb_onset_detect = detect_onset(mb_onset_strength)

    # map to grid
    strength_grid, onsets_grid = map_onsets_to_grid(grid_lines, mb_onset_strength, mb_onset_detect, n_fft=n_fft,
                                                    hop_length=hop_length, sr=sr)

    # concatenate in one single array
    mso = np.concatenate((strength_grid, onsets_grid), axis=1)

    return mso, spec, f_cq, strength_grid, onsets_grid, c_freq

def update_plot(plot1, figure, ax, data_mso, c_freq, inverty = False):
    if plot1 is None:
        plot1 = ax.imshow(data_mso.transpose(), cmap="twilight_shifted")
    else:
        plot1.set_data(data_mso.transpose())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    figure.colorbar(plot1, cax=cax, orientation='vertical')
    if inverty is True:
        ax.invert_yaxis()
    ax.set_ylabel("|--------------- Strength --------------||--------------- Utiming --------------|")
    figure.canvas.draw()
    figure.canvas.flush_events()
    return plot1

def prepare_plot (figsize=(15, 5)):
    figure, ax = plt.subplots(figsize=(15, 5))
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels([55, 90, 138, 175, 350, 6000, 8500, 12500, 55, 90, 138, 175, 350, 6000, 8500, 12500])

    plt.ion()
    plot1 = None
    return figure, ax, plot1

if __name__ == "__main__":
    figure, ax, plot1 = prepare_plot()

    filepath = "./tmp/internal_buffer_30_04_2022 14.19.57.wav"

    bpm = 120
    sr = 44100

    data, _ = mso_utils.read_input_wav(filepath, delete_after_read=False, sr=sr)

    grid_lines = mso_utils.create_grid_lines(bpm, sr, 33, start_before_0=True)

    data_mso, spec, f_cq, strength_grid, onsets_grid, c_freq= mso(data, grid_lines, sr=sr, reorder_to_start_before_gridline_0=True)

    for i in range(10):
        data_mso, spec, f_cq, strength_grid, onsets_grid, c_freq = mso(data, grid_lines, sr=sr,
                                                                       reorder_to_start_before_gridline_0=True)
        invert_y = True if i == 0 else False
        plot1 = update_plot(plot1, figure, ax, data_mso, c_freq, invert_y)
        plt.pause(1)
        print("i")

    plt.show()


