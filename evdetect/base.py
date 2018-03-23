from abc import abstractmethod


class EventModel:

    def __init__(self):
        self._x = None
        self._epsilon = None
        self._delta = None
        self._num_time_steps = None
        self._candidates = None
        self._reported_subsequences = None

    def init_detection(self, x, epsilon, delta):
        self._x = x
        self._epsilon = epsilon
        self._delta = delta
        self._num_time_steps = self._x.shape[0]
        self._candidates = set([])
        self._reported_subsequences = []

    def report_subsequences(self, t):
        for c in set(self._candidates):
            if self.should_report(c, t):
                length = c[2][0] - c[1][0] + 1
                likelihood = c[0] * self._epsilon ** length

                print('\n' + "Likelihood: {}".format(likelihood))
                print("Starting position: {} (in state {})".format(c[1][0], c[1][1]))
                print("End position: {} (in state {})".format(c[2][0], c[2][1]))

                self._reported_subsequences.append(c)
                self._candidates.remove(c)

    def end_detection(self):
        print('\n' + "Detection completed ({} subsequences found)".format(len(self._reported_subsequences)))
        reported_subsequences = self._reported_subsequences

        self._x = None
        self._epsilon = None
        self._delta = None
        self._num_time_steps = None
        self._candidates = None
        self._reported_subsequences = None

        return reported_subsequences

    @abstractmethod
    def should_report(self, c, t):
        pass

    @abstractmethod
    def detect_event(self, x, epsilon, delta):
        pass
