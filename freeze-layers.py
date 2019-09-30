from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Activation, Input, AveragePooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.models import Model
from keras.datasets import cifar10
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import Adam
from packaging import version
from PIL import Image
import numpy as np
import keras 

legacy_keras_with_bn_bug = version.parse(keras.__version__) < version.parse('2.1.3')

def resize(arr, shape):
  return np.array(Image.fromarray(arr).resize(shape))


num_classes = 10
reduce_train = 1000
reduce_test = 300
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test, y_train, y_test = x_train[:reduce_train], x_test[:reduce_test], y_train[:reduce_train], y_test[:reduce_test]
x_train = np.array([resize(x_train[i], (139, 139)) 
                          for i in range(0, len(x_train))]).astype('float32')
x_test = np.array([resize(x_test[i], (139, 139)) 
                          for i in range(0, len(x_test))]).astype('float32')
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(139, 139, 3))

for layer in resnet50.layers:
  layer.trainable = False
  if legacy_keras_with_bn_bug and isinstance(layer, BatchNormalization):
    layer._per_input_updates = {}

x = Conv2D(filters=100, kernel_size=2)(resnet50.output)
x = Dropout(0.4)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=resnet50.input, outputs=predictions)
model.summary()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), shuffle=True)