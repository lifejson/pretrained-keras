from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Activation, Input, AveragePooling2D
from keras.utils import to_categorical
from keras.models import Model
from keras.datasets import cifar10
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import Adam
from PIL import Image
import numpy as np

def resize(arr, shape):
  return np.array(Image.fromarray(arr).resize(shape))


num_classes = 10
reduce_train = 1000
reduce_test = 100
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
bottleneck_train_features = resnet50.predict(x_train)
bottleneck_test_features = resnet50.predict(x_test)


in_layer = Input(shape=(bottleneck_train_features.shape[1:]))
x = Conv2D(filters=100, kernel_size=2)(in_layer)
x = Dropout(0.4)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=in_layer, outputs=predictions)
model.summary()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
model.fit(bottleneck_train_features, y_train, batch_size=32, epochs=50, validation_data=(bottleneck_test_features, y_test), shuffle=True)