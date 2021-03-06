import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()  # setting random seed and default data type for numpy to use


# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# X = [[1], [2]]


##################################
# Long version of softmax function
##################################

# layer_outputs = [4.8, 1.21, 2.385]
#
# E = 2.71828182846
#
# exp_values = []
# for output in layer_outputs:
#     exp_values.append(E ** output)
# print('exponentiated values:')
# print(exp_values)
#
# norm_base = sum(exp_values) #sum all values
# norm_values = []
# for value in exp_values:
#     norm_values.append(value/norm_base)
# print('normalized expinentiated values:')
# print(norm_values)
# print('sim of normalized values: ', sum(norm_values))

###################################
# Sleek softmax function
###################################
# layer_outputs = [4.8, 1.21, 2.385]
#
# # For each value in a vector, calculate the exponential value
# exp_values = np.exp(layer_outputs)
# print('exponentiated values:')
# print(exp_values)
#
# # Now normalize values
# norm_values = exp_values / np.sum(exp_values)
# print('normalized exponentiated values:')
# print(norm_values)
#
# print('sum of normalized values: ', np.sum(norm_values))

##########################################

"""
sum(arrayOfArrays, axis=?, keepdims=?) ? = 0, 1 or none
0=columns(adds columns and outputs array)
1 = rows (adds rows and outputs array)
none = adds everything(outputs num)
keepdims = True || False - to keep original dimensions of array or not
"""
# layer_outputs = [[4.8, 1.21, 2.385],
#                  [8.9, -1.21, 0.2],
#                  [1.41, 1.051, 0.026]]
# print('so we can sum axis 1, but note the current shape:')
# print(np.sum(layer_outputs, axis=1, keepdims=True))

# X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # changes values so that is negative when exponentiated, so get fractional numbers

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second dense layer
# it takes output of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of the second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])


