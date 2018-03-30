"""
Detection of the G4-C5 sequence in a Bach piano piece
"""

import os

import numpy as np

import librosa

from evdetect.hmm import ConstrainedHiddenMarkovModel
from evdetect.utils import generate_spectrum, display_detection_result, detection_filter


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

    # instantiation of the constrained hidden Markov model
    model = ConstrainedHiddenMarkovModel(np.array([g4_spec, c5_spec]))

    # detection of the event's occurrences
    epsilon = 0.55
    delta = 1

    reported_subsequences = model.detect_event(x, epsilon, delta)

    # detection results
    display_detection_result(y, reported_subsequences, fs, hop_length)

    if not(os.path.exists('results')):
        os.makedirs('results')

    detection_filter(y, reported_subsequences, fs, hop_length, 'results/bach_filtered.wav')


pitch_sequence_detection()
