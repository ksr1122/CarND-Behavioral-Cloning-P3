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
