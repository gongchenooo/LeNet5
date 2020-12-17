import numpy as np

class MSE():
    def loss(y_true, y_pred):
        return np.sum(np.square(y_true - y_pred)) / 2.

    def derivative(y_true, y_pred):
        return y_pred - y_true

class LogLikelihood():
    def loss(y_true, y_pred):
        loss = np.sum(y_true * y_pred)
        loss = -np.log(loss) if loss != 0 else 500
        return loss
    
    def derivative(y_true, y_pred):
        d = y_pred.copy()
        d[np.argmax(y_true)] -= 1
        return d