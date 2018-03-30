"""
Detection of a bird's song in a noisy sound stage
"""

import numpy as np

import librosa
import librosa.display

from evdetect.hmm import HiddenMarkovModel
from evdetect.utils import import_annotations, display_detection_result, detection_filter


def birds():
    filename_annotations = '../examples/data/birds.lab'
    filename_audio = '../examples/data/birds.wav'

    # opening of the .wav file
    y, fs = librosa.load(filename_audio)
    spectrum = np.abs(librosa.stft(y))
    x = spectrum.transpose()

    hop_length = 2048 // 4

    # importation of the annotations
    annotations = import_annotations(filename_annotations)

    # instantiation of the hidden Markov model
    a = np.array([[0.5, 0.5], [0.5, 0.5]])
    pi = np.array([1.0, 0.0])
    cui_spec1 = np.random.random(x.shape[1])
    cui_spec2 = np.random.rand(x.shape[1])

    model = HiddenMarkovModel(a, pi, np.array([cui_spec1, cui_spec2]))

    # parameters learning
    x_train = [x[int(annotations[0] * fs / hop_length):int(annotations[-1] * fs / hop_length)]]
    n_iter = 1
    model.learn_parameters(x_train, n_iter)

    # detection of the event's occurrences
    epsilon = 0.35
    delta = 1

    reported_subsequences = model.detect_event(x, epsilon, delta)

    # detection results
    display_detection_result(y, reported_subsequences, fs, hop_length)
    detection_filter(y, reported_subsequences, fs, hop_length, 'results/birds_filtered.wav')


birds()
