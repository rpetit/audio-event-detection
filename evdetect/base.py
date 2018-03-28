"""
Base class for event models
"""

from abc import abstractmethod


class EventModel:
    """Abstract event model class"""

    def detect_event(self, x, epsilon, delta):
        """Detection interface

        Parameters
        ----------
        x : ndarray, shape (num_time_steps, num_freq_bins)
            The input amplitude spectrum
        epsilon : float
            Detection threshold
        delta : float
            Subsequence length penalization

        Returns
        -------
        list
            List of reported subsequences

        """
        assert(x.ndim == 2)

        self._init_detection(x, epsilon, delta)
        reported_subsequences = self._perform_detection()

        return reported_subsequences

    def _init_detection(self, x, epsilon, delta):
        """Initializes the detection

        Parameters
        ----------
        x : ndarray, shape (num_time_steps, num_freq_bins)
            The input amplitude spectrum
        epsilon : float
            Detection threshold
        delta : float
            Subsequence length penalization

        """
        self._x = x
        self._epsilon = epsilon
        self._delta = delta
        self._num_time_steps = self._x.shape[0]
        self._candidates = set([])
        self._reported_subsequences = []

    @abstractmethod
    def _perform_detection(self):
        """Main detection method

        Returns
        -------
        list
            List of reported subsequences

        """

    @abstractmethod
    def _should_report(self, c, t):
        """Whether a candidate subsequence should be reported at a given time or not

        Parameters
        ----------
        c : tuple
            Candidate subsequence
        t : int
            Current time index

        Returns
        -------
        bool
            True if c should be reported, False otherwise

        """
        pass

    def _report_subsequences(self, t):
        """Report all the subsequences that should be reported at a given time

        Parameters
        ----------
        t : int
            Current time index

        """
        for c in set(self._candidates):
            if self._should_report(c, t):
                length = c[2][0] - c[1][0] + 1
                likelihood = c[0] * self._epsilon ** length

                print('\n' + "Likelihood: {}".format(likelihood))
                print("Starting position: {} (in state {})".format(c[1][0], c[1][1]))
                print("End position: {} (in state {})".format(c[2][0], c[2][1]))

                self._reported_subsequences.append(c)
                self._candidates.remove(c)
