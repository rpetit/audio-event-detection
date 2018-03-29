import numpy as np

from evdetect.hmm import HiddenMarkovModel


a = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.9, 0, 0.1]])
pi = np.array([0.2, 0.8, 0])

ref_specs = np.array([[1, 1, 1], [-1, 0.1, 1], [0.1, 0.1, -1]]).astype('float64')

model = HiddenMarkovModel(a, ref_specs, pi)

x_train1 = np.array([[0.9, 0.9, 0.8], [-0.1, 0, -0.9], [-1, 0, 0.5]])
x_train2 = np.array([[0, 0.2, -0.5], [0, -0.1, -1.1], [1, 1, 1], [1, 0.9, 0.7]])

x_train = [x_train1, x_train2]

model.learn_parameters(x_train, 1)

print(model.a)
print(model.pi)
