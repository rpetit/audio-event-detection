import numpy as np
from scipy.stats import norm
import librosa

from event_detection import EventModel, detect_event

num_time_steps = 110
num_freq_bins = 100

filename = 'sound.wav'
y, sr = librosa.load(filename)
spectrum = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
spectrum = spectrum[:num_time_steps, :num_freq_bins]

do_spec = np.mean(spectrum[:, :37], axis=1)
mi_spec = np.mean(spectrum[:, 37:71], axis=1)
sol_spec = np.mean(spectrum[:, 71:110], axis=1)

emission_ref_specs = [do_spec]

amplitude_lower_bound = -80 * np.ones_like(emission_ref_specs)
amplitude_upper_bound = np.zeros_like(emission_ref_specs)
amplitude_variance = 100

x = spectrum.transpose()


def b(i, spec):
    cdf_lower_bound = norm.cdf(np.linalg.norm(amplitude_lower_bound - emission_ref_specs[i]) / amplitude_variance)
    cdf_upper_bound = norm.cdf(np.linalg.norm(amplitude_upper_bound - emission_ref_specs[i]) / amplitude_variance)
    normalizing_constant = amplitude_variance * (cdf_upper_bound - cdf_lower_bound)

    return norm.pdf(np.linalg.norm(spec - emission_ref_specs) / amplitude_variance) / normalizing_constant


model = EventModel(np.array([[1]]), b, np.array([1]))

detect_event(model, x, 1, 3)
