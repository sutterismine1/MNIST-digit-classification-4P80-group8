# COSC 4P80 Project
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Convolutional Layer Class

import random
import numpy as np

# Params for random initialization of Kenrel weights
RANGE = 1
BASE = 0
LOWEST_VAL = 0.001

class ConvolutionalLayer:

    def __init__(self, config, learning_rate, kernel_z):
        self.learning_rate = learning_rate

        assert 'kernel_count' in config, "\'kernel_count\' must be provided in a convolutional layer definition"
        self.kernel_count = config['kernel_count']

        assert 'kernel_dim' in config, "\'kernel_dim\' must be provided in a convolutional layer definition"
        self.kernel_shape = (config['kernel_dim'], config['kernel_dim'], kernel_z)
        self.kernel_x, self.kernel_y, self.kernel_z = self.kernel_shape

        assert 'padding' in config, "\'padding\' must be provided in a convolutional layer definition"
        self.padding = config['padding']

        assert 'pooling_dim' in config, "\'pooling_dim\' must be provided in a convolutional layer definition"
        self.pooling_dim = config['pooling_dim'] 

        self.kernels = [np.empty((self.kernel_z, self.kernel_x, self.kernel_y)) for _ in range(self.kernel_count)]  # z, x, y
        # randomly initialize weights
        for kernel in self.kernels:
            for i in range(len(kernel)): 
                for j in range(len(kernel[0])):
                    for k in range(len(kernel[0][0])):
                        val = random.random() * RANGE + BASE
                        while abs(val) < LOWEST_VAL:     # Guarentee starting weights are not too small
                            val = random.random() * RANGE + BASE

                        kernel[i][j][k] = val

        self.biases = [0.01 for _ in range(self.kernel_count)]

        self.first_layer = False

    # Feed x through the layer
    def apply(self, x):
        assert len(x) == self.kernel_z, "Number of channels does not match kernel_z"
        self.feature_maps = []

        # add padding to input
        self.input = np.pad(x, pad_width=((0, 0), (self.padding,self.padding),(self.padding,self.padding)), mode="constant", constant_values=0)

        # create feature_maps
        for i, kernel in enumerate(self.kernels):
            self.feature_maps.append(self.convolve_kernel(self.input, kernel) + self.biases[i])

        self.actvitated_maps = np.empty((len(self.feature_maps), len(self.feature_maps[0]), len(self.feature_maps[0][0])))
        self.d_activated_maps = np.empty((len(self.feature_maps), len(self.feature_maps[0]), len(self.feature_maps[0][0])))
        for layer in range(self.kernel_count):
            for i in range(len(self.feature_maps[0])):
                for j in range(len(self.feature_maps[0][0])):
                    self.actvitated_maps[layer][i][j] = self.__activate(i, j, layer)

        self.d_pool = np.zeros((len(self.feature_maps), len(self.feature_maps[0]), len(self.feature_maps[0][0])))
        # pool features
        self.pooled_maps = np.empty([
            self.kernel_count,                                      # z_dim of pooled maps                                                                        
            len(self.feature_maps[0]) // self.pooling_dim,       # x_dim of pooled maps
            len(self.feature_maps[0][0]) // self.pooling_dim    # y_dim of pooled maps
        ])
        
        for layer in range(self.kernel_count):
            for i in range(len(self.pooled_maps[0])):
                for j in range(len(self.pooled_maps[0][0])):
                    self.pooled_maps[layer][i][j] = self.__max_pool(i*self.pooling_dim, j*self.pooling_dim, layer)

        return self.pooled_maps

    # 3D-3D convolution -> 2D output map
    def convolve_kernel(self, x, kernel):
        feature_map = np.empty([
            len(x[0]) - len(kernel[0]) + 1,      # x_dim of feature maps
            len(x[0][0]) - len(kernel[0][0]) + 1     # y_dim of feature maps
        ])

        for channel in range(len(kernel)):
            for i in range(len(feature_map)):
                for j in range(len(feature_map[0])):
                    view = x[channel, i:i + len(kernel[0]), j:j + len(kernel[0][0])]
                    feature_map[i][j] = np.sum(view * kernel[channel]) # dot product of kernel and view

        return feature_map

    # ReLU
    # self.d_activated_maps is the derivative of the ReLU function
    # 1 for positive values, and 0 for negative values
    def __activate(self, i, j, layer):
        x = self.feature_maps[layer][i][j]
        self.d_activated_maps[layer][i][j] = -0.01 if x < 0 else 1
        return -0.01*x if x < 0 else x

    # Max Pool layer
    # self.d_pool keeps track of the derivative of the pooling layer. 
    # For max pooling, the winning position receives a 1, while the other positions are 0.
    def __max_pool(self, i, j, layer):
        maximum = float("-inf")
        max_i = 0
        max_j = 0
        for k in range(self.pooling_dim):
            for l in range(self.pooling_dim):
                val = self.actvitated_maps[layer][i+k][j+l]
                if val > maximum:
                    maximum = val
                    max_i = i+k
                    max_j = j+l

        self.d_pool[layer][max_i][max_j] = 1
        return maximum


    # !!! NEED TO FIX THIS !!!
    # Apply BP learning to convolutional layer
    # Based on:
    # https://www.youtube.com/watch?v=Pn7RK7tofPg
    # https://www.youtube.com/watch?v=vbUozbkMhI0
    def learn(self, prop_error):
        # dL/dP passed in as prop_error

        # calculate dL/dC using self.d_pool
        for layer in range(self.kernel_count):
            for i in range(len(prop_error[0])):
                for j in range(len(prop_error[0][0])):
                    self.d_pool[layer, i*self.pooling_dim:i*self.pooling_dim+self.pooling_dim, j*self.pooling_dim:j*self.pooling_dim+self.pooling_dim] *= prop_error[layer][i][j]

        # calculate dL/dZ
        # dL/dZ = dL/dC * dC/dZ
        # dL/dC is in self.d_pool
        # dC/dZ is in self.d_activated_maps
        dL_dZ = self.d_pool * self.d_activated_maps

        # Calculate error to propagate backwards (dL/dX)
        dL_dX = None
        if not self.first_layer:    # Don't calculate if you are the first layer, since no preceeding layer needs the propagated error
            dL_dX = np.zeros(self.input.shape)
            for k, kernel in enumerate(self.kernels):
                for channel in range(self.kernel_z):
                    # !!! I don't think this is right !!!
                    dL_dX[channel] += self.convolve_2D(np.pad(dL_dZ[k:k+1], ((0,0), (self.kernel_x-1, self.kernel_x-1), (self.kernel_y-1, self.kernel_y-1))), np.rot90(kernel[channel], 2))
        # Calculate weight and bias adjustments for each kernel
        for i, error in enumerate(dL_dZ):
            dL_dK = self.convolve_2D_outer(self.input, error)
            self.kernels[i] -= self.learning_rate * dL_dK # dL/dK
            self.biases[i] -= self.learning_rate * np.sum(error)  # dL/dB

        return dL_dX    # propagate error backwards

    # 3D-3D -> 3D output
    def convolve_3D(self, x, kernel):
        res = np.zeros([
            len(x),                                                 # z_dim
            len(x[0]) - len(kernel[0]) + 1,          # x_dim 
            len(x[0][0]) - len(kernel[0][0]) + 1     # y_dim
        ])

        for layer in range(len(res)):
            for i in range(len(res[0])):
                for j in range(len(res[0][0])):
                    view = x[layer, i:i + len(kernel[0]), j:j + len(kernel[0][0])]
                    res[layer][i][j] = np.sum(view * kernel[layer])

        return res

    # 3D-2D -> 3D output
    def convolve_2D_outer(self, x, kernel):
        res = np.zeros([
            len(x),                                                 # z_dim
            len(x[0]) - len(kernel) + 1,          # x_dim 
            len(x[0][0]) - len(kernel[0]) + 1     # y_dim
        ])

        for layer in range(len(res)):
            for i in range(len(res[0])):
                for j in range(len(res[0][0])):
                    view = x[layer, i:i + len(kernel), j:j + len(kernel[0])]
                    res[layer][i][j] = np.sum(view * kernel)

        return res

    # 3D-2D -> 2D output
    def convolve_2D(self, x, kernel):
        res = np.zeros([
            len(x[0]) - len(kernel) + 1,          # x_dim 
            len(x[0][0]) - len(kernel[0]) + 1     # y_dim
        ])

        for layer in range(len(x)):
            for i in range(len(res)):
                for j in range(len(res[0])):
                    view = x[layer, i:i + len(kernel), j:j + len(kernel[0])]
                    res[i][j] += np.sum(view * kernel)

        return res

    def set_first_layer(self, state):
        self.first_layer = state
