# COSC 4P80 Project Option 1: CNN from scrath to classify Mnist digits
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Run a trained model against the test set
# Data downloaded from https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data

from DataLoader import load_MNIST_data
from ConvolutionalNetwork import ConvolutionalNetwork
import json
import sys
import random
import numpy as np
import math

# Filepaths
input_path = './Data'
training_images_filepath = f'{input_path}/train-images.idx3-ubyte'
training_labels_filepath = f'{input_path}/train-labels.idx1-ubyte'
test_images_filepath = f'{input_path}/t10k-images.idx3-ubyte'
test_labels_filepath = f'{input_path}/t10k-labels.idx1-ubyte'

def main():
    # Read in arguments
    if len(sys.argv) < 2:
        print("Please provide the config file.")
        exit(1)

    # Read JSON config
    config_file = sys.argv[1]
    config = None
    with open(config_file) as json_file:
        config = json.load(json_file)

    # Seed PRNG
    assert 'trained_epochs' in config, "The config must be a trained network!"
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    print("Reading Data")
    (x_train, y_train), (x_test, y_test) = load_MNIST_data(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    x_train = np.array(x_train, dtype=float) / 255.0
    x_test  = np.array(x_test, dtype=float) / 255.0

    loss_function = calculate_error_CCE

    print("Initializing Network")
    network = ConvolutionalNetwork(config)

    global_error = 0

    print("Running Against Test Set")
    global_test_error = 0
    correct = 0
    class_count = 10  # magic number of 10 classes for digits 0-9, this line would need to be adjusted for different classification problems
    # tp, fp, fn are used for the precision and recall metrics where tp = true positive, fp = false positive, fn = false negative
    tp = np.zeros(class_count)
    fp = np.zeros(class_count)
    fn = np.zeros(class_count)
    for i in range(len(x_test)):
        # print out the iteration every 100 examples
        if i % 100 == 0:
            print(f"{i}/{len(x_test)}")
        image, digit = x_test[i], y_test[i]
        o = network.apply(image)

        global_test_error += loss_function(o, digit)

        confidence, prediction = find_winner(o)

        if prediction == digit:
            tp[digit]      += 1 # true positive for digit that was correctly guessed
        else:   # if prediction was incorrect
            fp[prediction] += 1 # false positive for digit that was incorrectly guessed
            fn[digit]      += 1 # flase negative for digit that should have been picked

        # print out the result of every test sample
        #print(f"Result: {prediction == digit}, Confidence: {confidence * 100}%, Output: {o}")

    correct = np.sum(tp) # correct predictions is equal to sum of true positives
    print(f"Test Correct: {correct / len(x_test) * 100}%, Global Test Error: {global_test_error / len(x_test)}")
    print(f"Test Recall: {tp/(tp+fn)*100}")
    print(f"Test Precision: {tp/(tp+fp)*100}")

# Mean Squared error across output vector
def calculate_error_MSE(o, y):
    expected_output = [0.0 for _ in range(len(o))]
    expected_output[y] = 1.0
    
    res = (o - expected_output)**2

    return np.mean(res)

# Categorical Cross-Entropy
def calculate_error_CCE(o, y):
    expected_output = np.zeros_like(o)
    expected_output[y] = 1.0

    clip_val = 1e-9
    o = np.clip(o, clip_val, 1.0 - clip_val) # avoid log(0) by clipping with a small value

    return -np.sum(expected_output * np.log(o))

# Take highest value as the winner
def find_winner(o):
    high = o[0]
    high_index = 0
    for i in range(len(o)):
        if o[i] > high:
            high = o[i]
            high_index = i

    return high, high_index

if __name__ == "__main__":
    main()