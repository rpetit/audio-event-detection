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
    pi : ndarray, shape (n_states,)
        Initial probabilities
    mu : ndarray, shape (n_states, n_freq_bins)
        Emission distributions parameters (Dirichlet)
    scaling : float
        Scaling factor used in emission density functions
    end_state : string
        Possible hidden end states, must be either 'all' or 'last'
    min_float : float
        Minimum float value used to avoid 0 values in log computations

    Returns
    -------
    list
        List of reported subsequences

    """

    def __init__(self, a, pi, mu, scaling=1.0, end_state='all', min_float=1e-50):
        assert(a.ndim == 2 and mu.ndim == 2 and pi.ndim == 1 and a.shape[0] == a.shape[1] == mu.shape[0] == pi.size)
        assert(np.all(np.isclose(np.sum(mu, axis=1), np.ones(mu.shape[0]))))

        self.a = a
        self.pi = pi
        self.mu = mu
        self.scaling = scaling

        self.log_a = np.log(self.a + min_float)
        self.log_pi = np.log(self.pi + min_float)

        self.n_states = self.pi.size

        assert(end_state == 'all' or end_state == 'last')
        self.end_state = end_state

    def log_b(self, i, spec):
        """Emission log density functions
        
        Parameters
        ----------
        i : int
            Index of the considered hidden state
        spec : ndarray, shape (n_freq_bins,) 

        Returns
        -------
        float
            Value (up to an additive constant) of the emission log density function associated with hidden state i
            evaluated at spec
        """
        normalized_spec = spec / np.sum(spec)
        return - self.scaling * np.sum(self.mu[i] * np.log(self.mu[i] / normalized_spec))

    def b(self, i, spec):
        """Emission density functions

        Parameters
        ----------
        i : int
            Index of the considered hidden state
        spec : ndarray, shape (n_freq_bins,)

        Returns
        -------
        float
            Value (up to a multiplicative constant) of the emission log density function associated with hidden state i
             evaluated at spec
        """
        normalized_spec = spec / np.sum(spec)
        return np.exp(- self.scaling * np.sum(self.mu[i] * np.log(self.mu[i] / normalized_spec)))

    def detect_event(self, x, epsilon, delta, display=False):
        """Event detection interface
        
        Parameters
        ----------
        x : ndarray, shape (n_steps, n_freq_bins)
            Input audio stream
        epsilon : float
            Likelihood threshold
        delta : float
            Minimum subsequence length
        display : bool
            Whether to display detection results or not

        Returns
        -------
        list
            List of reported subsequences

        """
        n_steps = x.shape[0]
        log_e = np.log(epsilon)
        
        log_v = np.zeros((n_steps, self.n_states))
        s = [[(-1, -1) for _ in range(self.n_states)] for _ in range(n_steps)]
        
        candidates = set([])
        reported_subsequences = []
        
        for t in range(n_steps):
            for i in range(self.n_states):
                if t == 0:
                    log_v[t, i] = self.log_pi[i] + self.log_b(i, x[t]) - log_e
                    s[t][i] = (t, i)
                else:
                    # log likelihood for starting the subsequence in (t, i)
                    log_v1 = self.log_pi[i] + self.log_b(i, x[t]) - log_e

                    # log likelihood for starting the subsequence before t
                    best_j = np.argmax(log_v[t - 1] + self.log_a[:, i])
                    log_v2 = log_v[t - 1][best_j] + self.log_a[best_j, i] + self.log_b(i, x[t]) - log_e

                    if log_v1 > log_v2:
                        s[t][i] = (t, i)
                        log_v[t, i] = log_v1
                    else:
                        s[t][i] = s[t - 1][best_j]
                        log_v[t, i] = log_v2

                if self.end_state == 'all' or i == self.n_states - 1:
                    if log_v[t, i] >= - delta * log_e:  # log likelihood threshold
                        # if the starting position is not shared with another candidate, i.e. Viterbi paths do not cross
                        if s[t][i] not in set([c[1] for c in candidates]):
                            candidates.add((log_v[t, i], s[t][i], (t, i)))
                        else:
                            for c in set(candidates):
                                # otherwise, keep the one with highest log likelihood
                                if s[t][i] == c[1] and log_v[t, i] >= c[0]:
                                    candidates.remove(c)
                                    candidates.add((log_v[t, i], s[t][i], (t, i)))
                
            for c in set(candidates):
                if c[1] not in set([s[t][i] for i in range(self.n_states)]) or t == n_steps - 1:
                    length = c[2][0] - c[1][0] + 1
                    log_likelihood = c[0] + length * log_e

                    if display:
                        print('\n' + "Log likelihood: {}".format(log_likelihood))
                        print("Starting position: {} (in state {})".format(c[1][0], c[1][1]))
                        print("End position: {} (in state {})".format(c[2][0], c[2][1]))

                    reported_subsequences.append((c[0], c[1][0], c[2][0]))
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

            likelihoods = []
            gamma = []
            xi = []

            # E step
            for k in range(num_train_seq):
                seq_likelihood, seq_gamma, seq_xi = self._forward_backward(x_train[k])

                likelihoods.append(seq_likelihood)
                gamma.append(seq_gamma)
                xi.append(seq_xi)

            # M step
            new_a = np.zeros_like(self.a)
            new_mu = np.zeros_like(self.mu)
            new_pi = np.zeros_like(self.pi)

            new_mu_norm = np.zeros(self.n_states)

            for k in range(num_train_seq):
                new_a += np.sum(xi[k] / likelihoods[k], axis=0)
                new_pi += gamma[k][0] / likelihoods[k]

                for i in range(self.n_states):
                    for t in range(x_train[k].shape[0]):
                        normalized_frame = x_train[k][t] / np.sum(x_train[k][t])
                        new_mu[i] += gamma[k][t, i] / likelihoods[k] * normalized_frame

                    new_mu_norm[i] += np.sum(gamma[k][:, i]) / likelihoods[k]

            new_pi *= 1 / np.sum(new_pi)

            for i in range(self.n_states):
                new_a[i] *= 1 / np.sum(new_a[i])
                new_mu[i] *= 1 / new_mu_norm[i]

            self.a = new_a
            self.mu = new_mu
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

        # forward variable recursion
        for t in range(n_steps):
            for i in range(self.n_states):
                if t == 0:
                    alpha[t, i] = self.pi[i] * self.b(i, x[t])
                else:
                    alpha[t, i] = np.sum(alpha[t - 1] * self.a[:, i] * self.b(i, x[t]))

        # backward variable recursion
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
