import numpy as np

from evdetect.base import EventModel


class HiddenMarkovModel(EventModel):

    def __init__(self, a, b, pi):
        self._a = a
        self._b = b
        self._pi = pi

        self._num_hidden_states = self._pi.size

        super(HiddenMarkovModel, self).__init__()

        self._v = None
        self._s = None

    def init_detection(self, x, epsilon, delta):
        super(HiddenMarkovModel, self).init_detection(x, epsilon, delta)

        self._v = np.zeros((self._num_time_steps, self._num_hidden_states))
        self._s = [[(-1, -1) for _ in range(self._num_hidden_states)] for _ in range(self._num_time_steps)]

    def end_detection(self):
        self._v = None
        self._s = None

        reported_subsequences = super(HiddenMarkovModel, self).end_detection()

        return reported_subsequences

    def compute_cumulative_likelihood(self, t, i):
        if t == 0:
            self._v[t, i] = self._pi[i] * self._b(i, self._x[t]) / self._epsilon
            self._s[t][i] = (t, i)
        else:
            v1 = self._pi[i] * self._b(i, self._x[t]) / self._epsilon

            best_j = np.argmax(self._v[t - 1] * self._a[:, i])
            v2 = self._v[t - 1][best_j] * self._a[best_j, i] * self._b(i, self._x[t]) / self._epsilon

            if v1 > v2:
                self._s[t][i] = (t, i)
                self._v[t, i] = v1
            else:
                self._s[t][i] = self._s[t - 1][best_j]
                self._v[t, i] = v2

    def update_candidates(self, t):
        for i in range(self._num_hidden_states):
            if self._v[t, i] >= 1 / self._epsilon ** self._delta:
                if self._s[t][i] not in set([c[1] for c in self._candidates]):
                    self._candidates.add((self._v[t, i], self._s[t][i], (t, i)))
                else:
                    for c in set(self._candidates):
                        if self._s[t][i] == c[1] and self._v[t, i] >= c[0]:
                            self._candidates.remove(c)
                            self._candidates.add((self._v[t, i], self._s[t][i], (t, i)))

    def should_report(self, c, t):
        return c[1] not in set([self._s[t][i] for i in range(self._num_hidden_states)]) or t == self._num_time_steps - 1

    def detect_event(self, x, epsilon, delta):
        self.init_detection(x, epsilon, delta)

        for t in range(self._num_time_steps):
            for i in range(self._num_hidden_states):
                self.compute_cumulative_likelihood(t, i)

            self.update_candidates(t)
            self.report_subsequences(t)

        reported_subsequences = self.end_detection()

        return reported_subsequences


class ConstrainedHiddenMarkovModel(HiddenMarkovModel):

    def __init__(self, num_hidden_states, b):
        a = np.zeros((num_hidden_states, num_hidden_states))

        for i in range(num_hidden_states - 1):
            a[i, i] = 0.5
            a[i, i + 1] = 0.5

        a[num_hidden_states - 1][num_hidden_states - 1] = 1

        pi = np.array([1] + [0] * (num_hidden_states - 1))

        super(ConstrainedHiddenMarkovModel, self).__init__(a, b, pi)

    def update_candidates(self, t):
        end_state = self._num_hidden_states - 1
        v_end_state = self._v[t, end_state]
        s_end_state = self._s[t][end_state]

        if v_end_state >= 1 / self._epsilon ** self._delta:
            if s_end_state not in set([c[1] for c in self._candidates]):
                self._candidates.add((v_end_state, s_end_state, (t, end_state)))
            else:
                for c in set(self._candidates):
                    if s_end_state == c[1] and v_end_state >= c[0]:
                        self._candidates.remove(c)
                        self._candidates.add((v_end_state, s_end_state, (t, end_state)))


class PitchSequenceModel(HiddenMarkovModel):

    def __init__(self, a, pi, pitch_specs, beta):

        def b(i, spec):
            normalized_spec = spec / np.sum(spec)
            return np.exp(- beta * pitch_specs[i] * np.log(pitch_specs[i] / normalized_spec).sum())

        super(PitchSequenceModel, self).__init__(a, b, pi)


class ConstrainedPitchSequenceModel(ConstrainedHiddenMarkovModel):

    def __init__(self, pitch_specs, beta):

        num_hidden_states = pitch_specs.shape[0]

        def b(i, spec):
            normalized_spec = spec / np.sum(spec)
            return np.exp(- beta * (pitch_specs[i] * np.log(pitch_specs[i] / normalized_spec)).sum())

        super(ConstrainedPitchSequenceModel, self).__init__(num_hidden_states, b)
