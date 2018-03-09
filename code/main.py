import numpy as np
import librosa

from event_models import SimplePitchModel, ComplexPitchModel, PitchSequenceModel
from spectrum_generation import generate_spectrum


def simple_pitch_detection(pitch='do'):
    filename = '../data/sound1.wav'
    y, sr = librosa.load(filename)
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = spectrum.shape[1]
    num_freq_bins = 20

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose()

    do_spec = np.mean(x[:37], axis=0)
    do_spec = do_spec / np.sum(do_spec)

    mi_spec = np.mean(x[37:71], axis=0)
    mi_spec = mi_spec / np.sum(mi_spec)

    sol_spec = np.mean(x[71:110], axis=0)
    sol_spec = sol_spec / np.sum(sol_spec)

    pitch_specs = {'do': do_spec, 'mi': mi_spec, 'sol': sol_spec}

    model = SimplePitchModel(pitch_specs[pitch], 10)
    model.detect_event(x, 0.01, 3)


def complex_pitch_detection():
    filename = '../data/sound2.wav'
    y, sr = librosa.load(filename)
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = spectrum.shape[1]
    num_freq_bins = 100

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose() + 1e-6

    attack_spec = np.mean(x[36:40], axis=0) + np.mean(x[105:109], axis=0)
    attack_spec += np.mean(x[179:183], axis=0) + np.mean(x[203:207], axis=0)
    attack_spec *= 1/4
    attack_spec = attack_spec / np.sum(attack_spec)

    sustain_spec = np.mean(x[40:104], axis=0) + np.mean(x[109:179], axis=0)
    sustain_spec += np.mean(x[183:203], axis=0) + np.mean(x[207:257], axis=0)
    sustain_spec *= 1/4
    sustain_spec = sustain_spec / np.sum(sustain_spec)

    model = ComplexPitchModel(attack_spec, sustain_spec, 20)

    model.detect_event(x, 0.19, 3)


def pitch_sequence_detection():
    filename = '../data/bach.mp3'
    y, sr = librosa.load(filename, duration=20)
    y = y[:441000]
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = spectrum.shape[1]
    num_freq_bins = 1024

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose() + 1e-6

    fs = 22050
    a0 = 1
    b = 0.8
    n_window = 1024
    n_fft = 2 * n_window

    g3_spec = generate_spectrum(392, fs, a0, b, n_window, n_fft)
    c4_spec = generate_spectrum(520, fs, a0, b, n_window, n_fft)

    model = PitchSequenceModel(np.array([g3_spec, c4_spec]), 2)

    model.detect_event(x, 1e-9, 1)


pitch_sequence_detection()
