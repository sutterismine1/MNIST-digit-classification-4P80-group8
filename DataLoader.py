# COSC 4P80 Project Option 1: CNN from scrath to classify Mnist digits
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Functions to load MNIST data

import numpy as np

def load_MNIST_data(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
    x_train, y_train = read_image_labels_set(training_images_filepath, training_labels_filepath)
    x_test, y_test = read_image_labels_set(test_images_filepath, test_labels_filepath)
    return (x_train, y_train), (x_test, y_test)

def read_image_labels_set(images_filepath, labels_filepath):
    images = None
    with open(images_filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big') # read big-endian magic number
        count = int.from_bytes(f.read(4), 'big') # read big-endian int for number of images
        row_size = int.from_bytes(f.read(4), 'big') # read big-endian row count
        col_size = int.from_bytes(f.read(4), 'big') # read big-endian column count
        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8).reshape((count, row_size, col_size))

    labels = None
    with open(labels_filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big') # read big-endian magic number
        count = int.from_bytes(f.read(4), 'big') # read big-endian int for number of images
        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)

    return images, labels



