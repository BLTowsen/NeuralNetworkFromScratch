import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()  # setting random seed and default data type for numpy to use


# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# X = [[1], [2]]


##################################
layer_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846

exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print('exponentiated values:')
print(exp_values)

norm_base = sum(exp_values) #sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value/norm_base)
print('normalized expinentiated values:')
print(norm_values)
print('sim of normalized values: ', sum(norm_values))


###################################



X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
print(X)

layer1.forward(X)
activation1.forward(layer1.output)

# print(layer1.output)

print(activation1.output)
