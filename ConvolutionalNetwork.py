# COSC 4P80 Project
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Convolutional Network Class
# Accepts a valid JSON config.
# Has 1+ convolutional layers (with ReLU and Max Pooling) and 0+ fully connected layers using Sigmoid and SoftMax

from ConvolutionalLayer import *
from FullyConnectedLayer import *
import numpy as np

class ConvolutionalNetwork:

    def __init__(self, config):
        assert 'learning_rate' in config, "The config must provide a value for \'learning_rate\'"
        self.learning_rate = config['learning_rate']

        assert 'convolutional_layers' in config, "The config must provide an array for \'convolutional_layers\'"
        assert len(config['convolutional_layers']) >= 1, "\'convolutional_layers\' must not be empty"
        
        prev_nodes = 28     # number of nodes in one dimension, hardcoded for this image but can make more general later
        kernel_z = 1        # depth of kernel. Starts as 1 for the single channel image
        self.convolutional_layers = []
        for conv_layer in config['convolutional_layers']:
            self.convolutional_layers.append(ConvolutionalLayer(conv_layer, self.learning_rate, kernel_z))
            kernel_z = conv_layer['kernel_count']
            prev_nodes = prev_nodes - conv_layer['kernel_dim'] + conv_layer['padding']*2 + 1
            prev_nodes = prev_nodes // conv_layer['pooling_dim']
        self.convolutional_layers[0].set_first_layer(True)

        self.last_convolutional_shape = (kernel_z, prev_nodes, prev_nodes)
        prev_nodes = kernel_z * prev_nodes**2
        self.fully_connected_layers = []
        if 'fully_connected_layers' in config and len(config['fully_connected_layers']) > 0:
            for fc_layer in config['fully_connected_layers']:
                assert 'node_count' in fc_layer, "The config must provide a value for \'node_count\' in each fully connected layer definition"
                self.fully_connected_layers.append(FullyConnectedLayer(self.learning_rate, prev_nodes, fc_layer['node_count'], hidden=True))
                prev_nodes = fc_layer['node_count']

        self.output_layer = FullyConnectedLayer(self.learning_rate, prev_nodes, 10, hidden=False) # add output layer, hardcoded 10 classes

    # Feed a sample x through the network, and apply backpropagation learning
    def apply_and_learn(self, x, y):
        o = self.apply(x)
        self.__learn(y)
        return o

    # Feed x through the network and receive o (output)
    def apply(self, x):
        channels = np.array([x]) # z, x, y
        # feed data through convolutional layers
        for conv_layer in self.convolutional_layers:
            channels = conv_layer.apply(channels)

        input_vector = channels.flatten()
        for layer in self.fully_connected_layers:
            input_vector = layer.apply(input_vector)

        return self.output_layer.apply(input_vector)

    # Apply backpropagation learning. Must be used after apply(x).
    def __learn(self, y):
        prop_error = self.output_layer.learn_output(y)

        for layer in reversed(self.fully_connected_layers):
            prop_error = layer.learn_hidden(prop_error)

        prop_error = prop_error.reshape(self.last_convolutional_shape)

        for layer in reversed(self.convolutional_layers):
            prop_error = layer.learn(prop_error)


        

