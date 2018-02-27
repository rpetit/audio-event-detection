import numpy as np
import librosa

from event_detection import PitchModel, detect_event

num_time_steps = 110
num_freq_bins = 100

filename = 'sound.wav'
y, sr = librosa.load(filename)
spectrum = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
spectrum = spectrum[:num_time_steps, :num_freq_bins]

do_spec = np.mean(spectrum[:, :37], axis=1)
mi_spec = np.mean(spectrum[:, 37:71], axis=1)
sol_spec = np.mean(spectrum[:, 71:110], axis=1)

amplitude_variance = 100

x = spectrum.transpose()

model = PitchModel(do_spec, amplitude_variance)
epsilon = 1
delta = 3

detect_event(model, x, epsilon, delta)
