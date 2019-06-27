#coding=utf-8

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        plt.savefig('/home/user045/fws/JQXX/Cifar10_classification/result/Loss_Acc_Curve.png')

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

num_classes = 10
               
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(256, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(256, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(Conv2D(256, (3, 3), activation=tf.nn.relu, padding='same', input_shape=train_images.shape[1:]))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

model.add(Dense(512, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation=tf.nn.softmax)) 

# compile the model
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy']) 

# train the model

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset 
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset 
        samplewise_std_normalization=False,  # divide each input by its std 
        zca_whitening=False,  # apply ZCA(Zero-phase Component Analysis) whitening 
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
 
datagen.fit(train_images)

history = LossHistory()

model.fit_generator(datagen.flow(train_images,train_labels,batch_size=32),epochs=200,validation_data=(test_images, test_labels),workers=4,callbacks=[history])

test_score = model.evaluate(test_images, test_labels,verbose=1)

# evaluate the model
print('Test_accuracy:', test_score[1])
history.loss_plot('epoch')