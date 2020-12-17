import numpy as np

class Convolution():
    def __init__(self, input_channels, filters, kernel=5, feature_mapping=None, activation='relu'):
        self.input_channels = input_channels
        self.filters = filters
        self.kernel = kernel
        self.feature_mapping = feature_mapping
        if self.feature_mapping is None:
            self.feature_mapping = np.ones((self.filters, self.input_channels))
        self.activation = activation
        self.init_parameters()

    def init_parameters(self):
        """
        w: 4D (filters, input_channels, width, height)
        b: 1D (filters)
        """
        self.W = np.random.randn(self.filters, self.input_channels, self.kernel, self.kernel) / np.sqrt(self.input_channels)
        # self.w = np.ones((self.filters, self.input_channels, self.kernel, self.kernel))
        for i in range(self.filters):
            for j in range(self.input_channels):
                if not self.feature_mapping[i][j]:
                    self.W[i][j] = 0
        self.b = np.zeros(self.filters)

    def convolution(self, input, filter, mapping=None):
        """
        input: 3D (channels, width, height)
        filter: 3D (channels, width, height)
        output: 2D (width, height)
        """
        dim = np.subtract(input[0].shape, filter[0].shape) + 1
        if mapping is None:
            mapping = np.ones(input.shape[0])
        output = np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                p = np.multiply(input[:, i:i+filter.shape[1], j:j+filter.shape[2]], filter).sum((1, 2))
                output[i][j] = np.sum(p * mapping)
        return output

    def pass_forward(self, input):
        """
        input: 3D (channels, width, height)
        output: 3D (filters, width, height)
        """
        self.input = input
        output = []
        for i in range(self.filters):
            self.z = self.convolution(input, self.W[i], self.feature_mapping[i]) + self.b[i]
            if self.activation == 'relu':
                output.append(np.maximum(0, self.z))
            else:
                output.append(self.z)
        self.output = np.array(output)
        return self.output

    def pass_backward(self, delta, eta):
        """
        delta: 3D (filters, width, height)
        output: 3D (input_channel, width, height)
        """
        if self.activation == 'relu':
            delta = delta * (self.z >= 0)
        delta_new = np.pad(delta, ((0,), (self.kernel-1,), (self.kernel-1,)), mode='constant', constant_values=0)
        delta_input = []
        for i in range(self.input_channels):
            W_j_i_rotate = np.array([np.rot90(np.rot90(self.W[j][i])) for j in range(self.filters)])
            delta_input.append(self.convolution(delta_new, W_j_i_rotate))
        delta_input = np.array(delta_input)

        for i in range(self.filters):
            delta_i = np.array([delta[i]])
            self.b[i] -= eta * np.sum(delta_i)
            for j in range(self.input_channels):
                if not self.feature_mapping[i][j]:
                    continue
                input_j = np.array([self.input[j]])
                delta_w = self.convolution(input_j, delta_i, self.feature_mapping[i][j])
                self.W[i][j] -= eta * delta_w

        return delta_input


class FullyConnect():
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.flat_input_size = np.product(input_size)
        self.output_size = output_size
        self.activation = activation
        self.init_parameters()

    def init_parameters(self):
        self.W = np.random.randn(self.flat_input_size, self.output_size) / np.sqrt(self.flat_input_size)
        # self.w = np.ones((self.output_size, self.flat_input_size))
        self.b = np.zeros(self.output_size)

    def pass_forward(self, input):
        self.input = input.reshape(self.flat_input_size)
        output = np.dot(self.input, self.W) + self.b
        if self.activation == 'relu':
            output = np.maximum(0, output)
        return output

    def pass_backward(self, delta, eta):
        delta_W = np.outer(self.input, delta)
        delta_b = delta
        delta_input = np.dot(delta, self.W.transpose())

        self.W -= eta * delta_W
        self.b -= eta * delta_b
        return delta_input.reshape(self.input_size)


class MaxPooling():
    def __init__(self, input_channels, size=2, stride=2):
        self.channels = input_channels
        self.size = size
        self.stride = stride

    def pass_forward(self, input):
        '''
        input: 3D (channels, width, height)
        '''
        input_w, input_h = input.shape[1:]
        output_w, output_h = input_w // self.stride, input_h // self.stride
        output = np.zeros((self.channels, output_w, output_h))
        self.pos = np.zeros((self.channels, output_w, output_h), dtype=np.int)
        for i in range(self.channels):
            for j in range(0, input_w, self.stride):
                for k in range(0, input_h, self.stride):
                    output[i, j//self.stride, k//self.stride] = np.max(input[i, j:j+self.size, k:k+self.size])
                    self.pos[i, j//self.stride, k//self.stride] = np.argmax(input[i, j:j+self.size, k:k+self.size])
        return output

    def pass_backward(self, delta):
        '''
        delta: 3D (channels, width, height)
        '''
        input_w, input_h = delta.shape[1:]
        output_w, output_h = input_w * self.stride, input_h * self.stride
        delta_input = np.zeros((self.channels, output_w, output_h))
        for i in range(self.channels):
            for j in range(0, input_w):
                for k in range(0, input_h):
                    delta_input[i, j*self.stride+self.pos[i, j, k]//self.size, k*self.stride+self.pos[i, j, k]%self.size] = delta[i, j, k]
        return delta_input

class Softmax():
    def __init__(self, size):
        self.size = size

    def pass_forward(self, input):
        self.input = input.reshape(self.size)
        e = np.exp(self.input - np.max(self.input))
        self.output = e / np.sum(e)
        return self.output

    def pass_backward(self, delta):
        predict = np.argmax(self.output)
        m = np.zeros((self.size, self.size))
        m[:, predict] = 1
        m = np.eye(self.size) - m
        d = np.diag(self.output) - np.outer(self.output, self.output)
        d = np.dot(delta, d)
        d = np.dot(d, m)
        return d