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
