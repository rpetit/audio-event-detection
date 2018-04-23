"""
Detection of a bird's song in a noisy sound stage
"""

import os

import numpy as np

import librosa
import librosa.display
from librosa.core import time_to_frames

from evdetect.hmm import HiddenMarkovModel
from evdetect.utils import import_annotations, display_detection_result, detection_filter


def bird():
    filename_annotations = '../examples/data/bird.lab'
    filename_audio = '../examples/data/bird.wav'
    filename_audio_filtered = '../examples/data/bird_filtered.wav'

    # opening of the .wav file
    y, fs = librosa.load(filename_audio)
    spectrum = np.abs(librosa.stft(y))
    x = spectrum.transpose()

    y_filtered, _ = librosa.load(filename_audio_filtered)
    spectrum_filtered = np.abs(librosa.stft(y_filtered))
    x_filtered = spectrum_filtered.transpose()

    hop_length = 2048 // 4

    # importation of the annotations
    annotations = import_annotations(filename_annotations)

    ref_spec1 = np.zeros(x.shape[1])
    ref_spec2 = np.zeros(x.shape[1])

    for k in range(len(annotations) - 1):
        if k % 2 == 0:
            ref_spec1 += np.mean(x_filtered[time_to_frames(annotations[k]):time_to_frames(annotations[k + 1])], axis=0)
        else:
            ref_spec2 += np.mean(x_filtered[time_to_frames(annotations[k]):time_to_frames(annotations[k + 1])], axis=0)

    ref_spec1 = ref_spec1 / np.sum(ref_spec1)
    ref_spec2 = ref_spec2 / np.sum(ref_spec2)

    # instantiation of the hidden Markov model
    a = np.array([[6/7, 1/7], [0.1, 0.9]])
    pi = np.array([1.0, 0.0])
    cui_spec1 = ref_spec1
    cui_spec2 = ref_spec2

    model = HiddenMarkovModel(a, pi, np.array([cui_spec1, cui_spec2]))

    # detection of the event's occurrences
    epsilon = 0.15
    delta = 1

    reported_subsequences = model.detect_event(x, epsilon, delta, display=True)

    # detection results
    display_detection_result(y, reported_subsequences, fs, hop_length)

    if not (os.path.exists('results')):
        os.makedirs('results')

    detection_filter(y, reported_subsequences, fs, hop_length, 'results/bird_hmm_filtered.wav')


bird()
