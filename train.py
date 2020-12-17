import numpy as np
from PIL import Image
from loss import MSE, LogLikelihood
from lenet5 import LeNet5
from data import load_mnist

epochs = 2
shuffle = True
lr = 0.001
def vectorize(num, classes):
    return [1 if i == num else 0 for i in range(classes)]
def main():
    # train_input, train_label, val_input, val_label, test_input, test_label = load_mnist()
    train_input = np.load('D:/3/AI/opencv-sudoku-solver/Data/English_train_data.npy')
    train_label = np.load('D:/3/AI/opencv-sudoku-solver/Data/English_train_labels.npy')
    test_input = np.load('D:/3/AI/opencv-sudoku-solver/Data/English_test_data.npy')
    test_label = np.load('D:/3/AI/opencv-sudoku-solver/Data/English_test_labels.npy')
    train_label = np.array([vectorize(test_label[i], 10) for i in train_label])
    test_label = np.array([vectorize(test_label[i], 10) for i in test_label])
    train_input = np.reshape(train_input, (60000, 1, 28, 28))
    test_input = np.reshape(test_input, (10000, 1, 28, 28))
    print(train_input.shape)
    print(test_input.shape)
    seq = np.arange(len(train_input))

    net = LeNet5(train_input[0].shape)
    for epoch in range(epochs):
        if shuffle: np.random.shuffle(seq)
        for step in range(len(train_input)):
            i = seq[step]
            x = train_input[i]
            y_true = train_label[i]
            y = net.forward(x)
            loss = LogLikelihood.loss(y_true, y)
            dloss = LogLikelihood.derivative(y_true, y)

            # print('Epoch %d step %d loss %f' % (epoch, step, loss))
            d = net.backward(dloss, lr)

            if step > 0 and step % 1000 == 0:
                net.save()
                correct = 0
                loss = 0
                #for i in range(len(test_input)):
                for i in range(200):
                    x = test_input[i]
                    y_true = test_label[i]
                    y = net.forward(x)
                    loss += LogLikelihood.loss(y_true, y)
                    if np.argmax(y) == np.argmax(y_true): correct += 1
                print('Validation accuracy: %.2f%%, average loss: %f' % (correct/200*100, loss/len(test_input)))
        correct = 0
        for i in range(len(test_input)):
            x = test_input[i]
            y_true = test_label[i]
            y = net.forward(x)
            if np.argmax(y) == np.argmax(y_true):
                correct += 1
        print('Test accuracy: %.2f%%' % (correct/len(test_input)*100))
if __name__ == '__main__':
    main()