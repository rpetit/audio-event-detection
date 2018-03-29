"""
Detection of a bird's song in a noisy sound stage
"""

import numpy as np

import librosa

from evdetect.hmm import HiddenMarkovModel
from evdetect.utils import import_annotations, export_subsequences


def birds():
    filename_annotations = '../examples/data/birds.lab'
    filename_audio = '../examples/data/birds.wav'

    # opening of the .wav file
    y, fs = librosa.load(filename_audio)
    spectrum = np.abs(librosa.stft(y))
    x = spectrum.transpose()

    hop_length = 2048 // 4

    # importation of the annotations and estimation of the reference spectra
    annotations = import_annotations(filename_annotations)

    cui_specs1 = []
    cui_specs2 = []

    for i in range(len(annotations) - 1):
        spec = x[int(annotations[i] * fs / hop_length):int(annotations[i + 1] * fs / hop_length)]
        if i % 2 == 0:
            cui_specs1.append(spec.mean(axis=0))
        else:
            cui_specs2.append(spec.mean(axis=0))

    # normalization of the reference spectra
    cui_spec1 = np.array(cui_specs1).mean(axis=0)
    cui_spec1 = cui_spec1 / np.sum(cui_spec1)
    cui_spec2 = np.array(cui_specs2).mean(axis=0)
    cui_spec2 = cui_spec2 / np.sum(cui_spec2)

    # instantiation of the hidden Markov model
    a = np.array([[0.5, 0.5], [0.5, 0.5]])
    pi = np.array([1.0, 0.0])

    model = HiddenMarkovModel(a, pi, np.array([cui_spec1, cui_spec2]))

    seq = [x[int(annotations[0] * fs / hop_length):int(annotations[-1] * fs / hop_length)]]
    model.learn_parameters(seq, 1)

    # detection of the event's occurrences
    epsilon = 0.35
    delta = 1

    reported_subsequences = model.detect_event(x, epsilon, delta)

    # exportation of the reported subsequences
    export_subsequences(reported_subsequences, fs, hop_length, '../examples/results/birds.lab')


birds()
