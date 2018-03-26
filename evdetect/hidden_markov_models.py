import numpy as np

from evdetect.base import EventModel


class HiddenMarkovModel(EventModel):
    """Hidden Markov model class

    Parameters
    ----------
    a : ndarray, shape (num_hidden_states, num_hidden_states)
        Transition probabilities
    b : function, parameters (int, ndarray)
        Emission density functions
    pi : ndarray, shape (num_hidden_states,)
        Initial probabilities
    """

    def __init__(self, a, b, pi):
        self._a = a
        self._b = b
        self._pi = pi

        self._num_hidden_states = self._pi.size

        super(HiddenMarkovModel, self).__init__()

        self._v = None  # cumulative likelihoods
        self._s = None  # starting positions

    def _init_detection(self, x, epsilon, delta):
        super(HiddenMarkovModel, self)._init_detection(x, epsilon, delta)

        self._v = np.zeros((self._num_time_steps, self._num_hidden_states))
        self._s = [[(-1, -1) for _ in range(self._num_hidden_states)] for _ in range(self._num_time_steps)]

    def _end_detection(self):
        self._v = None
        self._s = None

        super(HiddenMarkovModel, self)._end_detection()

    def _compute_cumulative_likelihood(self, t, i):
        """Cumulative likelihood computation (Viterbi-like recursion)

        Parameters
        ----------
        t : int
            Current time index
        i : int
            Hidden state index

        """
        if t == 0:
            self._v[t, i] = self._pi[i] * self._b(i, self._x[t]) / self._epsilon
            self._s[t][i] = (t, i)
        else:
            # likelihood for starting the subsequence in (t, i)
            v1 = self._pi[i] * self._b(i, self._x[t]) / self._epsilon

            # likelihood for starting the subsequence before t
            best_j = np.argmax(self._v[t - 1] * self._a[:, i])
            v2 = self._v[t - 1][best_j] * self._a[best_j, i] * self._b(i, self._x[t]) / self._epsilon

            if v1 > v2:
                self._s[t][i] = (t, i)
                self._v[t, i] = v1
            else:
                self._s[t][i] = self._s[t - 1][best_j]
                self._v[t, i] = v2

    def _update_candidates(self, t):
        """Candidates update

        Parameters
        ----------
        t : int
            Current time index

        """
        for i in range(self._num_hidden_states):
            if self._v[t, i] >= 1 / self._epsilon ** self._delta:  # likelihood threshold

                # if the starting position is not shared with another candidate (i.e. Viterbi paths do not cross)
                if self._s[t][i] not in set([c[1] for c in self._candidates]):
                    self._candidates.add((self._v[t, i], self._s[t][i], (t, i)))
                else:
                    for c in set(self._candidates):
                        # among the candidates sharing the same starting position, keep the one with highest likelihood
                        if self._s[t][i] == c[1] and self._v[t, i] >= c[0]:
                            self._candidates.remove(c)
                            self._candidates.add((self._v[t, i], self._s[t][i], (t, i)))

    def _should_report(self, c, t):
        # report if all optimal paths ending at t do not intersect with the candidates' path
        return c[1] not in set([self._s[t][i] for i in range(self._num_hidden_states)]) or t == self._num_time_steps - 1

    def _perform_detection(self):
        # Viterbi-like algorithm
        for t in range(self._num_time_steps):
            for i in range(self._num_hidden_states):
                self._compute_cumulative_likelihood(t, i)

            self._update_candidates(t)
            self._report_subsequences(t)

        return self._reported_subsequences


class ConstrainedHiddenMarkovModel(HiddenMarkovModel):
    """Constrained hidden markov model class

    Model in which the hidden process is constrained to end in the last hidden state

    Parameters
    ----------
    num_hidden_states : int
        Number of hidden states
    b : function, parameters (int, ndarray)
        Emission density functions

    """

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


class MarkovPitchSequenceModel(HiddenMarkovModel):
    """Markov pitch sequence model class

    Each hidden state represents a pitch within the sequence

    Parameters
    ----------
    a : ndarray, shape (num_hidden_states, num_hidden_states)
        Transition probabilities
    pitch_specs : ndarray, shape (num_hidden_states, num_freq_bins)
        Reference spectrum for each of the sequence's pitches
    beta : float
        Scaling factor used in the emission density functions

    """

    def __init__(self, a, pi, pitch_specs, beta):

        def b(i, spec):  # Dirichlet distribution
            normalized_spec = spec / np.sum(spec)
            return np.exp(- beta * (pitch_specs[i] * np.log(pitch_specs[i] / normalized_spec)).sum())

        super(MarkovPitchSequenceModel, self).__init__(a, b, pi)


class ConstrainedMarkovPitchSequenceModel(ConstrainedHiddenMarkovModel):
    """Constrained Markov pitch sequence model class

    Parameters
    ----------
    pitch_specs : ndarray, shape (num_hidden_states, num_freq_bins)
        Reference spectrum for each of the sequence's pitches
    beta : float
        Scaling factor used in the emission density functions

    """

    def __init__(self, pitch_specs, beta):

        num_hidden_states = pitch_specs.shape[0]

        def b(i, spec):  # Dirichlet distribution
            normalized_spec = spec / np.sum(spec)
            return np.exp(- beta * (pitch_specs[i] * np.log(pitch_specs[i] / normalized_spec)).sum())

        super(ConstrainedMarkovPitchSequenceModel, self).__init__(num_hidden_states, b)
