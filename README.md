# LeNet5
Simply run train.py to train the LeNet-5 Model
## `layers.py`
Include the implement of Convolution Layer(input also includes feature maps), FullyConnect Layer, MaxPooling layer, Softmax Layer(definition, pass forward, pass backward)
## `lenet5.py`
Convolution Layer1, MaxPooling Layer1, Convolution Layer2(based on feature maps), MaxPooling Layer2, FullyConnect Layer1, FullyConnect Layer2, FullyConnect Layer3, SoftmaxLayer
## `loss.py`
Include two kinds of loss: MSE(mean square error) and Loglikelihood
## `train.py`
Read data and train the LeNet-5 Net(need to modify the path of data)
