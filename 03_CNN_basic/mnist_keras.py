from gpu import  *
set_gpu()


import keras
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, Activation

batch_size = 128
num_classes = 10
epochs = 1

img_rows, img_cols = 28,28

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(64,kernel_size=(5,5), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(Dropout(0.025))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])