# COSC 4P80 Project
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Fully Connected Layer Class

import random
import numpy as np

# Params for random initialization
RANGE = 0.5
BASE = -0.25
LOWEST_VAL = 0.001

class FullyConnectedLayer:
    def __init__(self, learning_rate, input_nodes, output_nodes, hidden):
        self.learning_rate = learning_rate
        self.weights = np.empty([input_nodes, output_nodes])  # nodes[input_node][output_node]
        self.biases = np.empty(output_nodes)
        self.error = np.empty(output_nodes)
        self.hidden = hidden

        # randomly initialize weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                val = random.random() * RANGE + BASE
                while abs(val) < LOWEST_VAL:     # Guarentee starting weights are not too small
                    val = random.random() * RANGE + BASE

                self.weights[i][j] = val

        # randomly initialize biases
        for i in range(len(self.biases)):
            val = random.random() * RANGE + BASE
            while val < LOWEST_VAL:     # Guarentee starting biases are not too small
                val = random.random() * RANGE + BASE
            self.biases[i] = val

    # Feed an input vector x through this layer and return output vector
    def apply(self, x):
        self.output = np.empty(len(self.weights[0]))
        self.input = x
#        for i in range(len(self.weights[0])):
#            self.output[i] = self.__evaluate_node(i, x)
#
#        if not self.hidden: # output layer, apply softmax
#            self.output = self.softmax(self.output)
            
        if self.hidden: # hidden node, apply sigmoid
            for i in range(len(self.weights[0])):
                self.output[i] = self.__activate(self.__evaluate_node(i, x) + self.biases[i])
        else: # output layer, apply softmax
            for i in range(len(self.weights[0])):
                self.output[i] = self.__evaluate_node(i, x) + self.biases[i]
            self.output = self.softmax(self.output)

        return self.output

    # Evaluate a single neuron given input vector x
    def __evaluate_node(self, node, x):
        res = 0
        for i in range(len(x)):
            res += x[i] * self.weights[i][node]   # sum(input * weight)
        
        return res

    # activation function (sigmoid)
    def __activate(self, x):
        return 1 / (1 + np.exp(-x)) if x > -600 else 0  # avoid integer overflow from np.exp()

    # Adjust weights and biases of this Layer based on the current self.error values
    def __adjust_layer(self):
        self.weights -= self.learning_rate * np.outer(self.input, self.error)
        self.biases -= self.learning_rate * self.error

    # Backprop learning for the output layer
    # Based on slide 77 of Backpropagation slides
    def learn_output(self, expected_val):
        assert not self.hidden, "Trying to apply BP on the output layer with a hidden layer"
        
        expected_output = [0.0 for _ in range(len(self.output))]
        expected_output[expected_val] = 1.0

        # For Mean Squared Error (MSE)
        # derivative of sigmoid * derivative of mean squared error
        #self.error = self.output * (1 - self.output) * (self.output - expected_output)

        # For Categorical Cross-Entropy (CCE)
        # The derivative of softmax cancels out the denominator of Cross-Entropy derivative, simplifying down to y_hat - y
        self.error = self.output - expected_output
            
        next_error = np.zeros(len(self.weights))
        for i in range(len(self.weights)):
            next_error[i] = np.dot(self.weights[i], self.error)

        self.__adjust_layer()
            
        return next_error

    # Backprop learning for hidden layers
    # Based on slide 77 of Backpropagation slides
    def learn_hidden(self, prop_error):
        assert self.hidden, "Trying to apply BP on a hidden layer with an output layer"

        self.error = self.output * (1 - self.output) * prop_error

        next_error = np.zeros(len(self.weights))

        for i in range(len(self.weights)):
            next_error[i] = np.dot(self.weights[i], self.error)

        self.__adjust_layer()

        return next_error

    # convert data vector to softmax vector
    def softmax(self, data):
        z = data - np.max(data) # log-sum-exp trick to avoid overflow
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
