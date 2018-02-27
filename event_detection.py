import numpy as np


class EventModel:

    def __init__(self, a, b, pi):
        assert(a.ndim == 2 and pi.ndim == 1 and a.shape[0] == a.shape[1] == pi.size)

        self._a = a  # transition probabilities
        self._b = b  # emission density functions
        self._pi = pi  # initial probability

        self._num_hidden_states = self._pi.size

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def pi(self):
        return self._pi

    @property
    def num_hidden_states(self):
        return self._num_hidden_states


def detect_event(model, x, epsilon, delta):
    num_time_steps = x.shape[0]

    v = np.zeros((num_time_steps, model.num_hidden_states))  # cumulative likelihoods
    s = [[(0, 0) for _ in range(model.num_hidden_states)] for _ in range(num_time_steps)]  # starting positions

    candidates = set([])  # candidate sub-sequences

    for t in range(num_time_steps):
        for i in range(model.num_hidden_states):

            # cumulative likelihood computation
            if t == 0:
                v[t, i] = model.pi[i] * model.b(i, x[t]) / epsilon
            else:
                v1 = model.pi[i] * model.b(i, x[t]) / epsilon
                v2 = np.max(v[t - 1] * model.a[:, i]) * model.b(i, x[t]) / epsilon
                v[t, i] = max(v1, v2)

            # starting position computation
            if v[t, i] == model.pi[i] * model.b(i, x[t]) / epsilon:
                s[t][i] = (t, i)
            else:
                j = np.argmax(v[t - 1])
                s[t][i] = s[t - 1][j]

            # candidate sub-sequences update
            if v[t, i] >= 1 / epsilon ** delta:
                if s[t][i] not in set([c[1] for c in candidates]):
                    candidates.add((v[t, i], s[t][i], (t, i)))
                else:
                    for c in set(candidates):
                        if s[t][i] == c[1] and v[t, i] >= c[0]:
                            candidates.remove(c)
                            candidates.add((v[t, i], s[t][i], (t, i)))

        # optimal sub-sequence report
        for c in set(candidates):
            if c[1] not in set([s[t][i] for i in range(model.num_hidden_states)]) or t == num_time_steps - 1:
                length = c[2][0] - c[1][0] + 1
                likelihood = c[0] * epsilon ** length
                print("Cumulative likelihood: {}".format(likelihood))
                print("Starting position: {}".format(c[1][0]))
                print("End position: {}".format(c[2][0]))

                candidates.remove(c)
