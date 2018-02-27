import numpy as np
from scipy.stats import norm

import librosa

T = 110
freq_bins = 100

filename = 'sound.wav'
y, sr = librosa.load(filename)
S = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
S = S[:freq_bins, :T]
X = S.transpose()

do_spec = np.mean(S[:, :37], axis=1)
mi_spec = np.mean(S[:, 37:71], axis=1)
sol_spec = np.mean(S[:, 71:110], axis=1)

mu = sol_spec
A = -80 * np.ones_like(mu)
B = np.zeros_like(mu)
sigma = 100

epsilon = 1
delta = 3

v = np.zeros(T)
s = [0 for _ in range(T)]

S = set([])


def b(x):
    return norm.pdf(np.linalg.norm(x - mu) / sigma) / (sigma * (norm.cdf(np.linalg.norm(B - mu) / sigma)
                                                                - norm.cdf(np.linalg.norm(A - mu) / sigma)))


for t in range(T):

    # cumulative likelihood
    if t == 0:
        v[t] = b(X[t]) / epsilon
    else:
        v1 = b(X[t]) / epsilon
        v2 = v[t - 1] * b(X[t]) / epsilon
        v[t] = max(v1, v2)

        # starting position
        if v[t] == b(X[t]) / epsilon:
            s[t] = t
        else:
            s[t] = s[t - 1]

        if v[t] >= 1 / epsilon ** delta:
            if s[t] not in set([C[1] for C in S]):
                S.add((v[t], s[t], t))
            else:
                for C in set(S):
                    if s[t] == C[1] and v[t] >= C[0]:
                        S.remove(C)
                        S.add((v[t], s[t], t))

    for C in set(S):
        if C[1] != s[t] or t == T - 1:  # /!\ /!\ /!\ /!\
            length = C[2] - C[1] + 1
            likelihood = C[0] * epsilon ** length
            print("Likelihood: {}".format(likelihood))
            print("Starting position: {}".format(C[1]))
            print("End position: {}".format(C[2]))

            S.remove(C)