#!/usr/bin/env python3

import csv
import sklearn
import numpy as np

from math import ceil
from PIL import Image

DATA_PATH = './data/'

samples = []

with open(DATA_PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)    # skip header
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            def load_image(image_name, angle):
                image = np.asarray(Image.open(image_name))
                images.append(image)
                angles.append(angle)

                # flipped
                images.append(np.fliplr(image))
                angles.append(angle * -1.0)

            def load_batch(batch_sample):
                angle = float(batch_sample[3])
                load_image(DATA_PATH + batch_sample[0].strip(), angle)

                correction = 0.2

                angle_left = angle + correction
                load_image(DATA_PATH + batch_sample[1].strip(), angle_left)

                angle_right = angle - correction
                load_image(DATA_PATH + batch_sample[2].strip(), angle_right)
                
            for batch_sample in batch_samples:
                load_batch(batch_sample)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

y_size, x_size = 160, 320
crop_top, crop_bottom = 70, 25
row, col, ch = (y_size - crop_top - crop_bottom), x_size, 3  # Trimmed image format

from keras.models import Sequential, Model
from keras.layers import Conv2D, Cropping2D, Dense, Flatten, Lambda

def train():
    model = Sequential()

    # Cropping
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0,0)), input_shape=(y_size, x_size, ch)))
    # Normalize & Mean centere
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(row, col, ch), output_shape=(row, col, ch)))

    # NVIDIA's arch
    model.add(Conv2D(24,(5,5),activation='relu',strides=(2,2)))
    model.add(Conv2D(36,(5,5),activation='relu',strides=(2,2)))
    model.add(Conv2D(48,(5,5),activation='relu',strides=(2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(train_generator, \
                                         steps_per_epoch=ceil(len(train_samples)/batch_size), \
                                         validation_data=validation_generator, \
                                         validation_steps=ceil(len(validation_samples)/batch_size), \
                                         epochs=5, verbose=1)

    model.save('model.h5')
    return history_object