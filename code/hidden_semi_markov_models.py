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

        v = np.zeros((num_time_steps, self._num_hidden_states))  # cumulative likelihoods
        s = [[(0, 0) for _ in range(self._num_hidden_states)] for _ in range(num_time_steps)]  # starting positions

        candidates = set([])  # candidate sub-sequences

        for t in range(num_time_steps):
            for i in range(self._num_hidden_states):

                v1 = 0  # cumulative likelihood in case there is only one segment
                v2 = 0  # cumulative likelihood in case there are several segments
                emission_density_prod = 1  # product of the emission density values
                best_d1 = 0  # best duration (one segment case)
                best_d2 = 0  # best duration (several segments case)
                best_j2 = -1  # best previous state (several segments case)

                # loop over all possible durations for the subsequence's last segment
                for d in range(1, min(t + 2, 50)):

                    emission_density_prod *= self._b(i, x[t - d + 1])

                    temp_v1 = emission_density_prod * self._p(i, d) * self._pi[i] / epsilon ** d

                    if temp_v1 > v1 or d == 1:  # update or initial assignment
                        v1 = temp_v1
                        best_d1 = d

                    if d < t + 1:  # d = t + 1 is not possible since there is at least one previous segment
                        temp_best_j2 = np.argmax(v[t - d] * self._a[:, i])
                        temp_v2 = emission_density_prod * self._p(i, d) * v[t - d, temp_best_j2]
                        temp_v2 *= self._a[temp_best_j2, i] / epsilon ** d

                        if temp_v2 > v2 or d == 1:  # update or initial assignment
                            v2 = temp_v2
                            best_d2 = d
                            best_j2 = temp_best_j2

                # selection of the most likely configuration
                if v1 > v2:
                    v[t, i] = v1
                    s[t][i] = (t - best_d1 + 1, i)
                else:
                    v[t, i] = v2
                    s[t][i] = s[t - best_d2][best_j2]

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

        a = np.array([[0, 1], [0, 0]])
        pi = np.array([1, 0])

        # emission density functions (Dirichlet)
        def b(i, spec):
            normalized_spec = spec / np.sum(spec)
            return np.exp(- scaling_factor * (ref_specs[i] * np.log(ref_specs[i] / normalized_spec)).sum())

        # duration probability distributions (Poisson)
        def p(i, d):
            return mean_durations[i] ** d * np.exp(- mean_durations[i]) / gamma(d + 1)

        super(SemiMarkovPitchSequenceModel, self).__init__(a, b, p, pi)
