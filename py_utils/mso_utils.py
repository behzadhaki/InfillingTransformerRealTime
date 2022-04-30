import librosa
import os
import scipy.signal
import numpy as np
import math
import scipy.signal
import librosa
import warnings

def find_nearest(array, query):
    """
    Finds the closest entry in array to query. array must be sorted!
    @param array:                   a sorted array to search within
    @param query:                   value to find the nearest to
    @return index, array[index]:    the index in array closest to query, and the actual value
    """
    index = (np.abs(array-query)).argmin()
    return index, array[index]


def is_power_of_two(n):
    """
    Checks if a value is a power of two
    @param n:                               # value to check (must be int or float - otherwise assert error)
    @return:                                # True if a power of two, else false
    """
    if n is None:
        return False

    assert (isinstance(n, int) or isinstance(n, float)), "The value to check must be either int or float"

    if (isinstance(n, float) and n.is_integer()) or isinstance(n, int):
        # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        n = int(n)
        return (n & (n - 1) == 0) and n != 0
    else:
        return False


def read_input_wav(wav_filename, delete_after_read=True, sr=None):
    y, sr = librosa.load(wav_filename, sr) if sr is not None else librosa.load(wav_filename)
    if delete_after_read and os.path.exists(wav_filename):
        os.remove(wav_filename)
    return y, sr


#   -------------------------------------------------------------
#   Utils for computing the MSO::Multiband Synthesized Onsets
#   -------------------------------------------------------------

def cq_matrix(n_bins_per_octave, n_bins, f_min, n_fft, sr):
    """
    Constant-Q filterbank frequencies
    Based on https://github.com/mcartwright/dafx2018_adt/blob/master/large_vocab_adt_dafx2018/features.py
    @param n_bins_per_octave: int
    @param n_bins: int
    @param f_min: float
    @param n_fft: int
    @param sr: int
    @return c_mat: matrix
    @return: f_cq: list (triangular filters center frequencies)
    """
    # note range goes from -1 to bpo*num_oct for boundary issues
    f_cq = f_min * 2 ** ((np.arange(-1, n_bins + 1)) / n_bins_per_octave)  # center frequencies
    # centers in bins
    kc = np.round(f_cq * (n_fft / sr)).astype(int)
    c_mat = np.zeros([n_bins, int(np.round(n_fft / 2))])
    for k in range(1, kc.shape[0] - 1):
        l1 = kc[k] - kc[k - 1]
        w1 = scipy.signal.triang((l1 * 2) + 1)
        l2 = kc[k + 1] - kc[k]
        w2 = scipy.signal.triang((l2 * 2) + 1)
        wk = np.hstack(
            [w1[0:l1], w2[l2:]])  # concatenate two halves. l1 and l2 are different because of the log-spacing
        if (kc[k + 1] + 1) > c_mat.shape[1]: # if out of matrix shape, continue
            continue
        c_mat[k - 1, kc[k - 1]:(kc[k + 1] + 1)] = wk / np.sum(wk)  # normalized to unit sum;
    return c_mat, f_cq  # matrix with triangular filterbank

def logf_stft(x, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr):
    """
    Logf-stft
    Based on https://github.com/mcartwright/dafx2018_adt/blob/master/large_vocab_adt_dafx2018/features.py
    @param x: array
    @param n_fft: int
    @param win_length: int
    @param hop_length: int
    @param n_bins_per_octave: int
    @param n_octaves: int
    @param f_min: float
    @param sr: float. sample rate
    @return x_cq_spec: logf-stft
    """
    f_win = scipy.signal.hann(win_length)
    x_spec = librosa.stft(x,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=f_win)
    x_spec = np.abs(x_spec) / (2 * np.sum(f_win))

    # multiply stft by constant-q filterbank
    f_cq_mat, f_cq = cq_matrix(n_bins_per_octave, n_octaves * n_bins_per_octave, f_min, n_fft, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])
    stft = librosa.power_to_db(x_cq_spec).astype('float32')

    return stft, f_cq

def onset_strength_spec(x, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr, mean_filter_size):
    """
    Onset strength spectrogram
    Based on https://github.com/mcartwright/dafx2018_adt/blob/master/large_vocab_adt_dafx2018/features.py
    @param x: array
    @param n_fft: int
    @param win_length: int
    @param hop_length: int
    @param n_bins_per_octave: int
    @param n_octaves: int
    @param f_min: float
    @param sr: float. sample rate
    @param mean_filter_size: int. dt in the differential calculation
    @return od_fun: multi-band onset strength spectrogram
    @return f_cq: frequency bins of od_fun
    """

    f_win = scipy.signal.hann(win_length)
    x_spec = librosa.stft(x,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=f_win)
    x_spec = np.abs(x_spec) / (2 * np.sum(f_win))

    # multiply stft by constant-q filterbank
    f_cq_mat, f_cq = cq_matrix(n_bins_per_octave, n_octaves * n_bins_per_octave, f_min, n_fft, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])

    # subtract moving mean: difference between the current frame and the average of the previous mean_filter_size frames
    b = np.concatenate([[1], np.ones(mean_filter_size, dtype=float) / -mean_filter_size])
    od_fun = scipy.signal.lfilter(b, 1, x_cq_spec, axis=1)

    # half-wave rectify
    od_fun = np.maximum(0, od_fun)

    # post-process OPs
    od_fun = np.log10(1 + 1000 * od_fun)  ## log scaling
    od_fun = np.abs(od_fun).astype('float32')
    od_fun = np.moveaxis(od_fun, 1, 0)
    # clip
    # FIXME check value of 2.25
    od_fun = np.clip(od_fun / 2.25, 0, 1)

    return od_fun, f_cq

def reduce_f_bands_in_spec(freq_out, freq_in, S):
    """
    @param freq_out:        band center frequencies in output spectrogram
    @param freq_in:         band center frequencies in input spectrogram
    @param S:               spectrogram to reduce
    @returns S_out:         spectrogram reduced in frequency
    """

    if len(freq_out) >= len(freq_in):
        warnings.warn(
            "Number of bands in reduced spectrogram should be smaller than initial number of bands in spectrogram")

    n_timeframes = S.shape[0]
    n_bands = len(freq_out)

    # find index of closest input frequency
    freq_out_idx = np.array([], dtype=int)

    for f in freq_out:
        freq_out_idx = np.append(freq_out_idx, np.abs(freq_in - f).argmin())

    # band limits (not center)
    freq_out_band_idx = np.array([0], dtype=int)

    for i in range(len(freq_out_idx) - 1):
        li = np.ceil((freq_out_idx[i + 1] - freq_out_idx[i]) / 2) + freq_out_idx[i]  # find left border of band
        freq_out_band_idx = np.append(freq_out_band_idx, [li])

    freq_out_band_idx = np.append(freq_out_band_idx, len(freq_in))  # add last frequency in input spectrogram
    freq_out_band_idx = np.array(freq_out_band_idx, dtype=int)  # convert to int

    # init empty spectrogram
    S_out = np.zeros([n_timeframes, n_bands])

    # reduce spectrogram
    for i in range(len(freq_out_band_idx) - 1):
        li = freq_out_band_idx[i] + 1  # band left index
        if i == 0: li = 0
        ri = freq_out_band_idx[i + 1]  # band right index
        if li >= ri: # bands out of range
            S_out[:,i] = 0
        else:
            S_out[:, i] = np.max(S[:, li:ri], axis=1)  # pooling

    return S_out

def detect_onset(onset_strength):
    """
    Detects onset from onset strength envelope

    """
    n_timeframes = onset_strength.shape[0]
    n_bands = onset_strength.shape[1]

    onset_detect = np.zeros([n_timeframes, n_bands])

    for band in range(n_bands):
        time_frame_idx = librosa.onset.onset_detect(onset_envelope=onset_strength.T[band, :])
        onset_detect[time_frame_idx, band] = 1

    return onset_detect

def map_onsets_to_grid(grid, onset_strength, onset_detect, hop_length, n_fft, sr):
    """
    Maps matrices of onset strength and onset detection into a grid with a lower temporal resolution.
    @param grid:                 Array with timestamps
    @param onset_strength:       Matrix of onset strength values (n_timeframes x n_bands)
    @param onset_detect:         Matrix of onset detection (1,0) (n_timeframes x n_bands)
    @param hop_length:
    @param n_fft
    @return onsets_grid:         Onsets with respect to lines in grid (len_grid x n_bands)
    @return intensity_grid:      Strength values for each detected onset (len_grid x n_bands)
    """

    if onset_strength.shape != onset_detect.shape:
        warnings.warn(
            f"onset_strength shape and onset_detect shape must be equal. Instead, got {onset_strength.shape} and {onset_detect.shape}")

    n_bands = onset_strength.shape[1]
    n_timeframes = onset_detect.shape[0]
    n_timesteps = len(grid) - 1 # last grid line is first line of next bar

    # init intensity and onsets grid
    strength_grid = np.zeros([n_timesteps, n_bands])
    onsets_grid = np.zeros([n_timesteps, n_bands])

    # time array
    time = librosa.frames_to_time(np.arange(n_timeframes), sr=sr,
                                  hop_length=hop_length, n_fft=n_fft)

    #FIXME already defined in io_helpers. cannot be imported here because io_helpers has a HVO_Sequence() import
    def get_grid_position_and_utiming_in_hvo(start_time, grid):
        """
        Finds closes grid line and the utiming deviation from the grid for a queried onset time in sec

        @param start_time:                  Starting position of a note
        @param grid:                        Grid lines (list of time stamps in sec)
        @return tuple of grid_index,        the index of the grid line closes to note
                and utiming:                utiming ratio in (-0.5, 0.5) range
        """
        grid_index, grid_sec = find_nearest(grid, start_time)

        utiming = start_time - grid_sec  # utiming in sec

        if utiming < 0:  # Convert to a ratio between (-0.5, 0.5)
            if grid_index == 0:
                utiming = 0
            else:
                utiming = utiming / (grid[grid_index] - grid[grid_index - 1])
        else:
            if grid_index == (grid.shape[0] - 1):
                utiming = utiming / (grid[grid_index] - grid[grid_index - 1])
            else:
                utiming = utiming / (grid[grid_index + 1] - grid[grid_index])

        return grid_index, utiming

    # map onsets and strength into grid
    for band in range(n_bands):
        for timeframe_idx in range(n_timeframes):
            if onset_detect[timeframe_idx, band]:  # if there is an onset detected, get grid index and utiming
                grid_idx, utiming = get_grid_position_and_utiming_in_hvo(time[timeframe_idx], grid)
                if grid_idx == n_timesteps : continue # in case that a hit is assigned to last grid line
                strength_grid[grid_idx, band] = onset_strength[timeframe_idx, band]
                onsets_grid[grid_idx, band] = utiming

    return strength_grid, onsets_grid

def get_hvo_idxs_for_voice(voice_idx, n_voices):
    """
    Gets index for hits, velocity and offsets for a voice. Used for copying hvo values from a voice from an
    hvo_sequence to another one.
    """
    h_idx = voice_idx
    v_idx = [_ + n_voices for _ in voice_idx]
    o_idx = [_ + 2 * n_voices for _ in voice_idx]
    return h_idx, v_idx, o_idx

def get_logf_stft(y, **kwargs):
    sr = kwargs.get('sr', 44100)
    n_fft = kwargs.get('n_fft', 1024)
    win_length = kwargs.get('win_length', 1024)
    hop_length = kwargs.get('hop_length', 512)
    n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
    n_octaves = kwargs.get('n_octaves', 9)
    f_min = kwargs.get('f_min', 40)

    # get logstft for normalizaed audio
    mX, f_bins = logf_stft(y/np.max(np.abs(y)), n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr)

    return mX, f_bins

def get_onset_strength_spec(y, **kwargs):
    sr = kwargs.get('sr', 44100)
    n_fft = kwargs.get('n_fft', 1024)
    win_length = kwargs.get('win_length', 1024)
    hop_length = kwargs.get('hop_length', 512)
    n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
    n_octaves = kwargs.get('n_octaves', 9)
    f_min = kwargs.get('f_min', 40)
    mean_filter_size = kwargs.get('mean_filter_size', 22)

    # onset strength spectrogram for normalized audio
    spec, f_cq = onset_strength_spec(y/np.max(np.abs(y)), n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr,
                                     mean_filter_size)

    return spec, f_cq

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
        half_grid_res_in_samples = (grid_lines[1]-grid_lines[0]) * sr / 2
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

    return mso


def create_grid_lines(bpm, sr, n_lines, start_before_0):
    """
    creates grid lines (in sec)
    :param bpm: tempo
    :param sr:  sample rate
    :param start_before_0:  True    if signal starts at -.5 utiming of first grid line,
                            False   if signal starts at 0 utiming  of first grid line (starts on gridline[0])
    :return:
    """

    n_samples_qn = sr / bpm * 60            # number of samples in a single quarter note
    n_samples_16 = n_samples_qn / 4         # number of samples in a 16th  note
    n_samples_32 = n_samples_16 / 2         # number of samples in a 32nd  note

    grid_line_sample_indices = np.arange(n_lines) * n_samples_16
    grid_line_sample_indices = (grid_line_sample_indices + n_samples_32) \
        if start_before_0 is True else grid_line_sample_indices
    grid_lines_sec = grid_line_sample_indices / sr
    return grid_lines_sec



