import numpy as np
from scipy.special import gamma


class SegmentSequenceModel:

    def __init__(self, b, p, num_hidden_states):

        self._b = b  # emission density functions
        self._p = p  # duration probability distributions

        self._num_hidden_states = num_hidden_states

    def find_subsequences(self, x, epsilon, delta, max_segment_duration):
        optimal_subsequences = []

        num_time_steps = x.shape[0]  # number of time steps in the data

        # highest cumulative likelihoods (v[t, i] is the highest cumulative likelihood for being in state i at time t)
        v = np.zeros((num_time_steps, self._num_hidden_states))

        # starting positions (s[t][i] is the starting position corresponding to v[t, i])
        s = [[0 for _ in range(self._num_hidden_states)] for _ in range(num_time_steps)]

        candidates = set([])  # candidate sub-sequences

        for t in range(num_time_steps):
            for i in range(self._num_hidden_states):

                # computation of the highest cumulative likelihood and the starting position of hidden state i at time t

                emission_density_prod = 1  # product of the emission density values
                best_d = 0  # most likely segment duration

                for d in range(1, min(t + 2 - i, max_segment_duration)):  # loop over all possible segment durations
                    emission_density_prod *= self._b(i, x[t - d + 1])

                    if i == 0:  # this is the subsequence's first segment
                        temp_v = emission_density_prod * self._p(i, d) / epsilon ** d
                    else:  # there is at least one previous segment in the subsequence 
                        temp_v = emission_density_prod * self._p(i, d) * v[t - d, i - 1] / epsilon ** d

                    if temp_v >= v[t, i]:
                        v[t, i] = temp_v
                        best_d = d

                s[t][i] = (t - best_d + 1) if i == 0 else s[t - best_d][i - 1]

            # starting position and cumulative likelihood of the most likely subsequence ending at time t
            v_subsequence = v[t, self._num_hidden_states - 1]
            s_subsequence = s[t][self._num_hidden_states - 1]

            # add a candidate subsequence or update candidate subsequences
            if v_subsequence >= 1 / epsilon ** delta:
                if s_subsequence not in set([c[1] for c in candidates]):
                    candidates.add((v_subsequence, s_subsequence, t))
                else:
                    for c in set(candidates):
                        if s_subsequence == c[1] and v_subsequence >= c[0]:
                            candidates.remove(c)
                            candidates.add((v_subsequence, s_subsequence, t))

            # optimal sub-sequences report
            for c in set(candidates):
                if c[1] not in set([s[t][i] for i in range(2)]) or t == num_time_steps - 1:
                    length = c[2] - c[1] + 1
                    likelihood = c[0] * epsilon ** length

                    print('\n' + "Likelihood: {}".format(likelihood))
                    print("Starting position: {}".format(c[1]))
                    print("End position: {}".format(c[2]))

                    optimal_subsequences.append(c)
                    candidates.remove(c)

        print('\n' + "Detection completed ({} subsequences found)".format(len(optimal_subsequences)))

        return optimal_subsequences


class PitchSequenceModel(SegmentSequenceModel):

    def __init__(self, ref_specs, mean_durations, scaling_factor):

        spec_shapes = [spec.shape for spec in ref_specs]
        assert(np.all(spec_shapes == spec_shapes[0] * np.ones_like(spec_shapes)))
        assert(np.isclose(np.sum(ref_specs, axis=1), np.ones(ref_specs.shape[0])).all())

        num_hidden_states = ref_specs.shape[0]

        # emission density functions (Dirichlet)
        def b(i, spec):
            normalized_spec = spec / np.sum(spec)
            return np.exp(- scaling_factor * (ref_specs[i] * np.log(ref_specs[i] / normalized_spec)).sum())

        # duration probability distributions (Poisson)
        def p(i, d):
            return mean_durations[i] ** d * np.exp(- mean_durations[i]) / gamma(d + 1)

        super(PitchSequenceModel, self).__init__(b, p, num_hidden_states)
