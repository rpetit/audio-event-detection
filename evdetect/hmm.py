"""
Hidden Markov event models
"""

import numpy as np


class HiddenMarkovModel:
    """Hidden Markov model class

    Parameters
    ----------
    a : ndarray, shape (n_states, n_states)
        Transition probabilities
    b : function, parameters (int, ndarray)
        Emission density functions
    pi : ndarray, shape (n_states,)
        Initial probabilities
    end_states : string
        Possible hidden end states, must be either 'all' or 'last'

    Returns
    -------
    list
        List of reported subsequences

    """

    def __init__(self, a, b, pi, end_states='all'):
        assert(a.ndim == 2 and pi.ndim == 1 and a.shape[0] == a.shape[1] == pi.size)
        self.a = a
        self.b = b
        self.pi = pi

        self.n_states = self.pi.size

        assert(end_states == 'all' or end_states == 'last')
        self.end_states = end_states

    def detect_event(self, x, epsilon, delta):
        n_steps = x.shape[0]
        
        v = np.zeros((n_steps, self.n_states))
        s = [[(-1, -1) for _ in range(self.n_states)] for _ in range(n_steps)]
        
        candidates = set([])
        reported_subsequences = []
        
        for t in range(n_steps):
            for i in range(self.n_states):
                if t == 0:
                    v[t, i] = self.pi[i] * self.b(i, x[t]) / epsilon
                    s[t][i] = (t, i)
                else:
                    # likelihood for starting the subsequence in (t, i)
                    v1 = self.pi[i] * self.b(i, x[t]) / epsilon

                    # likelihood for starting the subsequence before t
                    best_j = np.argmax(v[t - 1] * self.a[:, i])
                    v2 = v[t - 1][best_j] * self.a[best_j, i] * self.b(i, x[t]) / epsilon

                    if v1 > v2:
                        s[t][i] = (t, i)
                        v[t, i] = v1
                    else:
                        s[t][i] = s[t - 1][best_j]
                        v[t, i] = v2

                if self.end_states == 'all' or i == self.n_states - 1:
                    if v[t, i] >= 1 / epsilon ** delta:  # likelihood threshold
                        # if the starting position is not shared with another candidate, i.e. Viterbi paths do not cross
                        if s[t][i] not in set([c[1] for c in candidates]):
                            candidates.add((v[t, i], s[t][i], (t, i)))
                        else:
                            for c in set(candidates):
                                # otherwise, keep the one with highest likelihood
                                if s[t][i] == c[1] and v[t, i] >= c[0]:
                                    candidates.remove(c)
                                    candidates.add((v[t, i], s[t][i], (t, i)))
                
            for c in set(candidates):
                if c[1] not in set([s[t][i] for i in range(self.n_states)]) or t == n_steps - 1:
                    length = c[2][0] - c[1][0] + 1
                    likelihood = c[0] * epsilon ** length

                    print('\n' + "Likelihood: {}".format(likelihood))
                    print("Starting position: {} (in state {})".format(c[1][0], c[1][1]))
                    print("End position: {} (in state {})".format(c[2][0], c[2][1]))

                    reported_subsequences.append(c)
                    candidates.remove(c)

        return reported_subsequences

    def learn_parameters(self, x_train, n_iter):
        """Parameters learning (EM algorithm) interface

        Parameters
        ----------
        x_train : list
            List of training sequences
        n_iter : int
            Number of EM iterations to perform

        """
        num_train_seq = len(x_train)
        
        for _ in range(n_iter):

            seq_likelihoods = []
            gamma = []
            xi = []

            # E step
            for k in range(num_train_seq):
                seq_likelihood, gamma_k, xi_k = self._forward_backward(x_train[k])

                seq_likelihoods.append(seq_likelihood)
                gamma.append(gamma_k)
                xi.append(xi_k)

            # M step
            # TODO: deal with emission parameters
            new_a = np.zeros((self.n_states, self.n_states))
            new_pi = np.zeros(self.n_states)

            for i in range(self.n_states):
                for j in range(self.n_states):
                    for k in range(num_train_seq):
                        new_a[i, j] += 1 / seq_likelihoods[k] * np.sum(xi[k][1:, i, j])

                new_a[i] *= 1 / np.sum(new_a[i])

            for k in range(num_train_seq):
                new_pi += 1 / seq_likelihoods[k] * gamma[k][0]

            new_pi *= 1 / np.sum(new_pi)

            self.a = new_a
            self.pi = new_pi

    def _forward_backward(self, x):
        """Forward backward recursion

        Parameters
        ----------
        x : ndarray, shape (n_steps, n_freq_bins)

        Returns
        -------
        tuple
            (seq_likelihood, gamma, xi) with seq_likelihood the marginal likelihood of x under the model

        """
        n_steps = x.shape[0]

        alpha = np.zeros((n_steps, self.n_states))
        beta = np.zeros((n_steps, self.n_states))
        gamma = np.zeros((n_steps, self.n_states))
        xi = np.zeros((n_steps, self.n_states, self.n_states))

        for t in range(n_steps):
            for i in range(self.n_states):
                if t == 0:
                    alpha[t, i] = self.pi[i] * self.b(i, x[t])
                else:
                    alpha[t, i] = np.sum(alpha[t - 1] * self.a[:, i] * self.b(i, x[t]))

        for t in reversed(range(n_steps)):
            for i in range(self.n_states):
                if t == n_steps - 1:
                    beta[t, i] = 1
                else:
                    emissions = np.array([self.b(j, x[t]) for j in range(self.n_states)])
                    beta[t, i] = np.sum(self.a[i, :] * emissions * beta[t + 1])

        for t in range(n_steps):
            gamma[t] = alpha[t] * beta[t] / np.sum(alpha[t] * beta[t])

            if t > 0:
                for j in range(self.n_states):
                    xi[t, :, j] = alpha[t - 1] * self.a[:, j] * self.b(j, x[t]) * beta[t, j]

                xi[t] *= 1 / np.sum(xi[t])

        seq_likelihood = np.sum(alpha[-1])

        return seq_likelihood, gamma, xi


class ConstrainedHiddenMarkovModel(HiddenMarkovModel):
    """Constrained hidden Markov model class

    Model in which the hidden process is constrained TODO: add details

    Parameters
    ----------
    n_states : int
        Number of hidden states
    b : function, parameters (int, ndarray)
        Emission density functions

    """

    def __init__(self, n_states, b):
        a = np.zeros((n_states, n_states))

        for i in range(n_states - 1):
            a[i, i] = 0.5
            a[i, i + 1] = 0.5

        a[n_states - 1][n_states - 1] = 1

        pi = np.array([1] + [0] * (n_states - 1))

        super(ConstrainedHiddenMarkovModel, self).__init__(a, b, pi, end_states='last')


class MarkovPitchSequenceModel(HiddenMarkovModel):
    """Markov pitch sequence model class

    Each hidden state represents a pitch within the sequence

    Parameters
    ----------
    a : ndarray, shape (n_states, n_states)
        Transition probabilities
    pitch_specs : ndarray, shape (n_states, num_freq_bins)
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
    pitch_specs : ndarray, shape (n_states, num_freq_bins)
        Reference spectrum for each of the sequence's pitches
    beta : float
        Scaling factor used in the emission density functions

    """

    def __init__(self, pitch_specs, beta):

        n_states = pitch_specs.shape[0]

        def b(i, spec):  # Dirichlet distribution
            normalized_spec = spec / np.sum(spec)
            return np.exp(- beta * (pitch_specs[i] * np.log(pitch_specs[i] / normalized_spec)).sum())

        super(ConstrainedMarkovPitchSequenceModel, self).__init__(n_states, b)
