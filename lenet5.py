import numpy as np
from layers import Convolution
from layers import FullyConnect
from layers import MaxPooling
from layers import Softmax

class LeNet5():
    def __init__(self, input_size):
        '''
        input_size: 3D (channels, width, height)
        '''
        self.input_size = input_size
        self.conv1 = Convolution(input_size[0], 6)
        self.maxpool1 = MaxPooling(6)
        self.conv2 = Convolution(6, 16, feature_mapping=[
            [1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1]
        ])
        self.maxpool2 = MaxPooling(16)
        self.fc1 = FullyConnect((16, 4, 4), 120)
        self.fc2 = FullyConnect(120, 84)
        self.fc3 = FullyConnect(84, 10)
        self.softmax = Softmax(10)

    def forward(self, x):
        out = self.conv1.pass_forward(x)
        out = self.maxpool1.pass_forward(out)
        out = self.conv2.pass_forward(out)
        out = self.maxpool2.pass_forward(out)
        out = self.fc1.pass_forward(out)
        out = self.fc2.pass_forward(out)
        out = self.fc3.pass_forward(out)
        out = self.softmax.pass_forward(out)
        return out

    def backward(self, delta, eta):
        out = self.softmax.pass_backward(delta)
        out = self.fc3.pass_backward(out, eta)
        out = self.fc2.pass_backward(out, eta)
        out = self.fc1.pass_backward(out, eta)
        out = self.maxpool2.pass_backward(out)
        out = self.conv2.pass_backward(out, eta)
        out = self.maxpool1.pass_backward(out)
        out = self.conv1.pass_backward(out, eta)
        return out

    def save(self):
        dic = {'conv1.W': self.conv1.W, 'conv1.b': self.conv1.b,
               'conv2.W': self.conv2.W, 'conv2.b': self.conv2.b,
               'fc1.W': self.fc1.W, 'fc1.b': self.fc1.b,
               'fc2.W': self.fc2.W, 'fc2.b': self.fc2.b,
               'fc3.W': self.fc3.W, 'fc3.b': self.fc3.b}
        np.save('parameters.npy', dic)