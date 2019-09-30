from keras import backend as K
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Taken from official cifar: https://www.cs.toronto.edu/~kriz/cifar.html
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

sample_num = 4
fig = plt.figure(figsize=(12, 12))
for i in range(sample_num):
  fig.add_subplot(1, sample_num, i+1).title.set_text(f'Shape: {x_train[i].shape}')
  plt.imshow(x_train[i])

print(np.take(labels, y_train[:sample_num]))