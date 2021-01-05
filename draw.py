import matplotlib.pyplot as plt
import numpy as np
acc = np.load('accuracy.npy')
epochs = range(len(acc))  # Get number of epochs
    # 画accuracy曲线
plt.plot(epochs, acc, 'r')
plt.title('Test Accuracy of LeNet-5')
plt.xlabel("Steps(per 3000)")
plt.ylabel("Accuracy")
fig1 = plt.gcf()
fig1.savefig('Test Accuracy of LeNet-5.png', dpi=300)
plt.figure()

loss = np.load('loss.npy')
plt.plot(epochs, loss, 'b')
plt.title('Test Loss of LeNet-5')
plt.xlabel("Steps(per 3000)")
plt.ylabel("Loss")
fig1 = plt.gcf()
fig1.savefig('Test Loss of LeNet-5.png', dpi=300)
plt.figure()