import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift


def generate_spectrum(f0, fs, a0, b, n_window, n_fft):
    hann_window = signal.hann(n_window)
    hann_window_fft = fft(hann_window, n_fft) / (len(hann_window) / 2.0)
    hann_window_response = np.abs(fftshift(hann_window_fft) / abs(hann_window_fft).max())

    n_harmonics = fs // (2 * f0)
    harmonic_spec = np.zeros(n_fft)
    for n in range(1, n_harmonics):
        harmonic_spec[int(round(n_fft * n * f0 / fs))] = a0 * np.exp(-b * n)

    for n in range(1, n_harmonics):
        harmonic_spec[harmonic_spec.size - n] = harmonic_spec[n]

    harmonic_spec_window_conv = signal.convolve(hann_window_response, fftshift(harmonic_spec))
    shift = fftshift(harmonic_spec_window_conv)
    output_spec = shift[:n_fft // 2]

    return output_spec
