import numpy as np
from scipy.special import gamma


# TODO: set a maximum duration for segments
class HiddenSemiMarkovModel:

    def __init__(self, a, b, p, pi):
        assert(a.ndim == 2 and pi.ndim == 1 and a.shape[0] == a.shape[1] == pi.size)

        self._a = a  # transition probabilities
        self._b = b  # emission density functions
        self._p = p  # duration probability distributions
        self._pi = pi  # initial probability

        self._num_hidden_states = self._pi.size

    def find_subsequences(self, x, epsilon, delta):
        optimal_subsequences = []

        num_time_steps = x.shape[0]

        v = np.zeros((num_time_steps, self._num_hidden_states))  # cumulative likelihoods (segment end)
        s = [[(0, 0) for _ in range(self._num_hidden_states)] for _ in range(num_time_steps)]  # starting positions

        candidates = set([])  # candidate sub-sequences

        for t in range(num_time_steps):
            for i in range(self._num_hidden_states):

                # cumulative likelihood and starting position computation
                if t == 0:
                    v[t, i] = self._pi[i] * self._b(i, x[t]) * self._p(i, 1) / epsilon
                    s[t][i] = (t, i)
                else:
                    # cumulative likelihood in case the subsequence starts at (t, i)
                    v1 = self._pi[i] * self._b(i, x[t]) * self._p(i, 1)

                    # cumulative likelihood in case the subsequence starts before (t, i)
                    v2 = 0
                    obs_likelihood = 1
                    best_d = 0
                    best_j = -1

                    # loop over all possible durations for the subsequence's last segment
                    for d in range(1, t + 2):
                        obs_likelihood *= self._b(i, x[t - d + 1])
                        temp_best_j = np.argmax(v[t - d] * self._a[:, i])
                        segment_likelihood = obs_likelihood * self._p(i, d) * v[t - d, temp_best_j]
                        segment_likelihood *= self._a[temp_best_j, i]

                        if v2 < segment_likelihood:
                            v2 = segment_likelihood
                            best_d = d
                            best_j = temp_best_j

                    # selection of the most likely configuration
                    if v1 > v2:
                        s[t][i] = (t, i)
                        v[t, i] = v1 / epsilon
                    else:
                        s[t][i] = s[t - best_d][best_j]
                        v[t, i] = v2 / epsilon ** best_d

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
                if c[1] not in set([s[t][i] for i in range(self._num_hidden_states)]) or t == num_time_steps - 1:
                    length = c[2][0] - c[1][0] + 1
                    likelihood = c[0] * epsilon ** length

                    print('\n' + "Likelihood: {}".format(likelihood))
                    print("Starting position: {} (in state {})".format(c[1][0], c[1][1]))
                    print("End position: {} (in state {})".format(c[2][0], c[2][1]))

                    optimal_subsequences.append(c)
                    candidates.remove(c)

        print('\n' + "Detection completed ({} subsequences found)".format(len(optimal_subsequences)))

        return optimal_subsequences


class SemiMarkovPitchSequenceModel(HiddenSemiMarkovModel):

    def __init__(self, ref_specs, mean_durations, scaling_factor):

        spec_shapes = [spec.shape for spec in ref_specs]
        assert(np.all(spec_shapes == spec_shapes[0] * np.ones_like(spec_shapes)))
        assert(np.isclose(np.sum(ref_specs, axis=1), np.ones(ref_specs.shape[0])).all())

        a = np.array([[0, 1], [0, 1]])
        pi = np.array([1, 0])

        # emission density functions (Dirichlet)
        def b(i, spec):
            normalized_spec = spec / np.sum(spec)
            return np.exp(- scaling_factor * (ref_specs[i] * np.log(ref_specs[i] / normalized_spec)).sum())

        # duration probability distributions (Poisson)
        def p(i, d):
            return np.exp(mean_durations[i] ** d * np.exp(- mean_durations[i]) / gamma(d + 1)) if d < 100 else 0

        super(SemiMarkovPitchSequenceModel, self).__init__(a, b, p, pi)
