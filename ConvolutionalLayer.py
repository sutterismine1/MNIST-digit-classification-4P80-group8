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
                        #kernel[i][j][k] = np.random.uniform(-0.25, 0.25)


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

        self.feature_maps = np.array(self.feature_maps)

        self.activated_maps = np.empty((len(self.feature_maps), len(self.feature_maps[0]), len(self.feature_maps[0][0])))
        self.d_activated_maps = np.empty((len(self.feature_maps), len(self.feature_maps[0]), len(self.feature_maps[0][0])))
        '''
        for layer in range(self.kernel_count):
            for i in range(len(self.feature_maps[0])):
                for j in range(len(self.feature_maps[0][0])):
                    self.activated_maps[layer][i][j] = self.__activate(i, j, layer)
        '''
        # vectorized ReLu
        self.activated_maps = np.maximum(0.0, self.feature_maps)
        self.d_activated_maps = (self.feature_maps > 0.0).astype(float)

        self.d_pool = np.zeros((len(self.feature_maps), len(self.feature_maps[0]), len(self.feature_maps[0][0])))
        # using a boolean array instead of a number array to hold the max.
        self.d_pool_switches = np.zeros_like(self.activated_maps, dtype=bool)

        # pool features
        self.pooled_maps = np.empty([
            self.kernel_count,                                      # z_dim of pooled maps                                                                        
            self.feature_maps.shape[1] // self.pooling_dim,       # x_dim of pooled maps
            self.feature_maps.shape[2] // self.pooling_dim    # y_dim of pooled maps
        ])
        
        for layer in range(self.kernel_count):
            for i in range(len(self.pooled_maps[0])):
                for j in range(len(self.pooled_maps[0][0])):
                    self.pooled_maps[layer][i][j] = self.__max_pool(i*self.pooling_dim, j*self.pooling_dim, layer)

        return self.pooled_maps

    # 3D-3D convolution -> 2D output map
    def convolve_kernel(self, x, kernel):
        
        z, x_dim, y_dim = x.shape
        kx, ky = kernel.shape[1], kernel.shape[2]
        
        out_x = x_dim - kx + 1
        out_y = y_dim - ky + 1
        
        feature_map = np.zeros((
            out_x,      # x_dim of feature maps
            out_y     # y_dim of feature maps
        ))

        #for channel in range(len(kernel)):
            #for i in range(out_x):
                #for j in range(out_y):
                    #view = x[channel, i:i + len(kernel[0]), j:j + len(kernel[0][0])]
                    #feature_map[i][j] += np.sum(view * kernel[channel]) # dot product of kernel and view

        for i in range(out_x):
            for j in range(out_y):
                patch = x[:, i:i+kx, j:j+ky]
                feature_map[i, j] = np.tensordot(patch, kernel, axes=([0,1,2],[0,1,2]))

        return feature_map

    # ReLU
    # self.d_activated_maps is the derivative of the ReLU function
    # 1 for positive values, and 0 for negative values
    def __activate(self, i, j, layer):
        x = self.feature_maps[layer][i][j]
        self.d_activated_maps[layer][i][j] = 0 if x < 0 else 1
        return 0 if x < 0 else x

    '''
    # Max Pool layer
    # self.d_pool keeps track of the derivative of the pooling layer. 
    # For max pooling, the winning position receives a 1, while the other positions are 0.
    def __max_pool(self, i, j, layer):
        maximum = float("-inf")
        max_i = 0
        max_j = 0
        for k in range(self.pooling_dim):
            for l in range(self.pooling_dim):
                val = self.activated_maps[layer][i+k][j+l]
                if val > maximum:
                    maximum = val
                    max_i = i+k
                    max_j = j+l

            self.d_pool[layer][max_i][max_j] = 1
            return maximum
    '''

    # Max Pool layer
    # self.d_pool keeps track of the derivative of the pooling layer. 
    # For max pooling, the winning position receives a true, while the other positions are defalt false.
    # using activated maps instead of feature_maps
    def __max_pool(self, i, j, layer):
        maximum = float("-inf")
        max_i = 0
        max_j = 0
        for k in range(self.pooling_dim):
            for l in range(self.pooling_dim):
                val = self.activated_maps[layer][i+k][j+l]
                if val > maximum:
                    maximum = val
                    max_i = i+k
                    max_j = j+l

        self.d_pool_switches[layer][max_i][max_j] = True
        return maximum

    # !!! NEED TO FIX THIS !!!
    # Apply BP learning to convolutional layer
    # Based on:
    # https://www.youtube.com/watch?v=Pn7RK7tofPg
    # https://www.youtube.com/watch?v=vbUozbkMhI0
    def learn(self, prop_error):
        # dL/dP passed in as prop_error
        '''
        # calculate dL/dC using self.d_pool
        for layer in range(self.kernel_count):
            for i in range(len(prop_error[0])):
                for j in range(len(prop_error[0][0])):
                    self.d_pool[
                    layer, 
                    i*self.pooling_dim:i*self.pooling_dim+self.pooling_dim, 
                    j*self.pooling_dim:j*self.pooling_dim+self.pooling_dim
                    ] *= prop_error[layer][i][j]
        
        # calculate dL/dZ
        # dL/dZ = dL/dC * dC/dZ
        # dL/dC is in self.d_pool
        # dC/dZ is in self.d_activated_maps
        dL_dZ = self.d_pool * self.d_activated_maps
        
        '''
        # calculate dL/dC using d_pool_switches instead
        dL_dC = np.zeros_like(self.activated_maps)

        for layer in range(self.kernel_count):
            for i in range(len(prop_error[0])):
                for j in range(len(prop_error[0][0])):
                    dL_dC[
                        layer,
                        i*self.pooling_dim:(i+1)*self.pooling_dim,
                        j*self.pooling_dim:(j+1)*self.pooling_dim
                    ] += (
                        self.d_pool_switches[
                            layer,
                            i*self.pooling_dim:(i+1)*self.pooling_dim,
                            j*self.pooling_dim:(j+1)*self.pooling_dim
                        ] * prop_error[layer, i, j]
                    )
        
        # calculate dL/dZ
        # dL/dZ = dL/dC * dC/dZ
        dL_dZ = dL_dC * self.d_activated_maps

        '''
        # Calculate error to propagate backwards (dL/dX)
        dL_dX = None
        if not self.first_layer:    # Don't calculate if you are the first layer, since no preceeding layer needs the propagated error
            dL_dX = np.zeros(self.input.shape)
            for k, kernel in enumerate(self.kernels):
                for channel in range(self.kernel_z):
                    # !!! I don't think this is right !!!
                    dL_dX[channel] += self.convolve_2D(
                    np.pad(
                    dL_dZ[k:k+1], 
                    ((0,0), (self.kernel_x - 1, self.kernel_x - 1), 
                    (self.kernel_y - 1, self.kernel_y - 1))), 
                    np.rot90(kernel[channel], 2))
        '''

        # Calculate error to propagate backwards (dL/dX)
        dL_dX = None
        if not self.first_layer:
            dL_dX = np.zeros_like(self.input)

            for k in range(self.kernel_count):
                for channel in range(self.kernel_z):
                    dL_dX[channel] += self.convolve_2D(
                        np.pad(
                            dL_dZ[k],
                            ((self.kernel_x - 1, self.kernel_x - 1),
                            (self.kernel_y - 1, self.kernel_y - 1)),
                            mode="constant"
                        ),
                        np.rot90(self.kernels[k][channel], 2)
                    )
        
        '''
        # Calculate weight and bias adjustments for each kernel
        for i, error in enumerate(dL_dZ):
            dL_dK = self.convolve_2D_outer(self.input, error)
            self.kernels[i] -= self.learning_rate * dL_dK # dL/dK
            self.biases[i] -= self.learning_rate * np.sum(error)  # dL/dB
        '''

        # Calculate weight and bias adjustments for each kernel
        for k in range(self.kernel_count):
            dL_dK = self.convolve_2D_outer(self.input, dL_dZ[k])
            self.kernels[k] -= self.learning_rate * dL_dK
            self.biases[k] -= self.learning_rate * np.mean(dL_dZ[k])
            '''
            for z in range(self.kernel_z):
                for x in range(self.kernel_x):
                    for y in range(self.kernel_y):
                        self.kernels[k][z, x, y] -= self.learning_rate * np.sum(
                            self.input[z,
                                    x:x + dL_dZ.shape[1],
                                    y:y + dL_dZ.shape[2]] 
                                    * dL_dZ[k]
                        )

            # bias update
            self.biases[k] -= self.learning_rate * np.mean(dL_dZ[k])
            '''

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
            len(x),                               # z_dim
            len(x[0]) - len(kernel) + 1,          # x_dim 
            len(x[0][0]) - len(kernel[0]) + 1     # y_dim
        ])

        for layer in range(len(res)):
            for i in range(len(res[0])):
                for j in range(len(res[0][0])):
                    view = x[layer, i:i + len(kernel), j:j + len(kernel[0])]
                    res[layer][i][j] = np.sum(view * kernel)

        return res
    
    '''
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
    '''
    
    # 3D-2D -> 2D output
    def convolve_2D(self, x, kernel):
        # if x is 2D, add a channel dimension
        if x.ndim == 2:
            x = x[np.newaxis, :, :]

        kx, ky = kernel.shape
        res = np.zeros((
            x.shape[1] - kx + 1,   # x_dim
            x.shape[2] - ky + 1    # y_dim
        ))

        for layer in range(x.shape[0]):
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    view = x[layer, i:i+kx, j:j+ky]
                    res[i, j] += np.sum(view * kernel)

        return res


    def set_first_layer(self, state):
        self.first_layer = state
