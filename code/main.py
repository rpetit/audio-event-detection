import numpy as np
import librosa

from utils import generate_spectrum, export_subsequences
from event_models import SimplePitchModel, ComplexPitchModel, PitchSequenceModel


def simple_pitch_detection(pitch='do'):
    filename = '../data/sound1.wav'
    y, fs = librosa.load(filename)
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = 110
    num_freq_bins = 1024

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose()

    a0 = 1
    b = 0.8
    n_window = 1024
    n_fft = 2 * n_window

    c3_spec = generate_spectrum(132, fs, a0, b, n_window, n_fft)
    e3_spec = generate_spectrum(165, fs, a0, b, n_window, n_fft)
    g3_spec = generate_spectrum(198, fs, a0, b, n_window, n_fft)
    pitch_specs = {'do': c3_spec, 'mi': e3_spec, 'sol': g3_spec}

    mean_duration = 40
    scaling_factor = 1.4
    model = SimplePitchModel(pitch_specs[pitch], mean_duration, scaling_factor)

    epsilon = 0.052
    delta = 3
    max_segment_length = 60
    optimal_subsequences = model.find_subsequences(x, epsilon, delta, max_segment_length)
    export_subsequences(optimal_subsequences, fs, n_fft // 4, '../results/simple_pitch.lab')


def complex_pitch_detection():
    filename = '../data/sound2.wav'
    y, fs = librosa.load(filename)
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = spectrum.shape[1]
    num_freq_bins = 100

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose() + 1e-6

    n_fft = 2048

    attack_spec = np.mean(x[36:40], axis=0) + np.mean(x[105:109], axis=0)
    attack_spec += np.mean(x[179:183], axis=0) + np.mean(x[203:207], axis=0)
    attack_spec *= 1/4
    attack_spec = attack_spec / np.sum(attack_spec)

    sustain_spec = np.mean(x[40:104], axis=0) + np.mean(x[109:179], axis=0)
    sustain_spec += np.mean(x[183:203], axis=0) + np.mean(x[207:257], axis=0)
    sustain_spec *= 1/4
    sustain_spec = sustain_spec / np.sum(sustain_spec)

    attack_mean_duration = 4
    sustain_mean_duration = 40
    scaling_factor = 1.4
    model = ComplexPitchModel(attack_spec, sustain_spec, attack_mean_duration, sustain_mean_duration, scaling_factor)

    epsilon = 0.8
    delta = 3
    max_segment_length = 60
    optimal_subsequences = model.find_subsequences(x, epsilon, delta, max_segment_length)
    export_subsequences(optimal_subsequences, fs, n_fft // 4, '../results/complex_pitch.lab')


def pitch_sequence_detection():
    filename = '../data/bach.wav'
    y, fs = librosa.load(filename, sr=22050, duration=20)
    y = y[:fs * 20]
    spectrum = np.abs(librosa.stft(y))

    num_time_steps = spectrum.shape[1]
    num_freq_bins = 1024

    spectrum = spectrum[:num_freq_bins, :num_time_steps]
    x = spectrum.transpose() + 1e-6

    a0 = 1
    b = 0.8
    n_window = 1024
    n_fft = 2 * n_window

    g4_spec = generate_spectrum(392, fs, a0, b, n_window, n_fft)
    c5_spec = generate_spectrum(522, fs, a0, b, n_window, n_fft)

    mean_durations = [8, 8]
    scaling_factor = 1.4

    model = PitchSequenceModel(np.array([g4_spec, c5_spec]), mean_durations, scaling_factor)

    epsilon = 0.09
    delta = 1
    max_segment_length = 50

    optimal_subsequences = model.find_subsequences(x, epsilon, delta, max_segment_length)
    export_subsequences(optimal_subsequences, fs, n_fft // 4, '../results/pitch_sequence.lab')


complex_pitch_detection()
