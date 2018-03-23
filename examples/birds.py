import numpy as np

import librosa

from evdetect.hidden_markov_models import PitchSequenceModel
from utils import import_annotations, export_subsequences


def birds():
    filename_filtered = '../examples/data/birds_filtered.wav'
    filename_annotations = '../examples/data/birds_filtered.lab'
    filename = '../examples/data/birds.wav'

    y, fs = librosa.load(filename)
    y_filtered, fs_filtered = librosa.load(filename_filtered)

    assert(fs == fs_filtered)

    annotations = import_annotations(filename_annotations)

    print(annotations)

    # /!\ work in progress /!\


birds()
