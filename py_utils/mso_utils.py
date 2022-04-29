import librosa
import os

def read_input_wav(wav_filename, delete_after_read=True, sr=None):
    y, sr = librosa.load(wav_filename, sr) if sr is not None else librosa.load(wav_filename)
    if delete_after_read and os.path.exists(wav_filename):
        os.remove(wav_filename)
    return y, sr

