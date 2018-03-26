"""
Detection of the G4-C5 sequence in a Bach piano piece
"""

import numpy as np

import librosa

from evdetect.hidden_markov_models import ConstrainedMarkovPitchSequenceModel
from evdetect.utils import generate_spectrum, export_subsequences


def pitch_sequence_detection():
    # opening the .wav audio file
    filename = '../examples/data/bach.wav'
    y, fs = librosa.load(filename, sr=22050, duration=20)
    y = y[:fs * 20]
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = spectrum.shape[1]
    num_freq_bins = 1024

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose() + 1e-6

    # reference spectrum generation
    a0 = 1
    b = 0.8
    n_window = 1024
    n_fft = 2 * n_window
    hop_length = n_fft // 4

    g4_spec = generate_spectrum(392, fs, a0, b, n_window, n_fft)
    c5_spec = generate_spectrum(522, fs, a0, b, n_window, n_fft)

    # instantiation of the constrained Markov pitch sequence model
    scaling_factor = 1.4

    model = ConstrainedMarkovPitchSequenceModel(np.array([g4_spec, c5_spec]), scaling_factor)

    # detection of the event's occurrences
    epsilon = 0.07
    delta = 1

    reported_subsequences = model.detect_event(x, epsilon, delta)

    # exportation of the reported subsequences
    export_subsequences(reported_subsequences, fs, hop_length, '../examples/results/pitch_sequence.lab')


pitch_sequence_detection()
