# COSC 4P80 Project Option 1: CNN from scrath to classify Mnist digits
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Data downloaded from https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data

from MnistDataloader import MnistDataloader
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
    if len(sys.argv) < 1:
        print("Please provide the config file.")
        exit(1)

    # Read JSON config
    config_file = sys.argv[1]
    config = None
    with open(config_file) as json_file:
        config = json.load(json_file)

    # Seed PRNG
    assert 'seed' in config, "The config must provide a value for \'seed\'"
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    assert 'epochs' in config, "The config must provide a value for \'epochs\'"
    epochs = config['epochs']

    assert 'output_filename' in config, "The config must provide a value for \'output_filename\'"
    output_filename = config['output_filename']

    print("Reading Data")
    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            for k in range(len(x_train[i][j])):
                x_train[i][j][k] = x_train[i][j][k] / 255
    for i in range(len(x_test)):
        for j in range(len(x_test[i])):
            for k in range(len(x_test[i][j])):
                x_test[i][j][k] = x_test[i][j][k] / 255

    x_train = x_train[:10000] # REMOVE AFTER, just doing 500 samples to test correctness of learning algorithm implementation
    y_train = y_train[:10000] # REMOVE AFTER
    x_test = x_test[:2000]
    y_test = y_test[:2000]

    # Create an ordering that will be shuffled between epochs
    data_map = [x for x in range(len(x_train))]

    loss_function = calculate_error_CCE         # CHANGE LOSS FUNCTION HERE AND IN learn_output() OF FullyConnectedLayer.py

    print("Initializing Network")
    network = ConvolutionalNetwork(config)

    output_file = open(output_filename+".csv", "w")
    output_file.write("Epoch, Training Set, Test Set\n")
    output_file.flush()

    print("Starting Training")
    step = 0    # used to keep track of how many samples have been trained on
    for epoch in range(1, epochs+1):
        global_error = 0
        for index in data_map:
            o = network.apply_and_learn(x_train[index], y_train[index])

            global_error += loss_function(o, y_train[index])
            step += 1

            if step % 200 == 0 and step != 0:
                print(step)

        # shuffle data_map
        for i in range(math.floor(len(x_train) * 0.20)):
            r1, r2 = random.randint(0, len(x_train)-1), random.randint(0, len(x_train)-1)
            data_map[r1], data_map[r2] = data_map[r2], data_map[r1]

        

        global_test_error = 0
        correct = 0
        for i in range(len(x_test)):
            image, digit = x_test[i], y_test[i]   # Just testing on the training set until the CNN is correct
            o = network.apply(image)

            global_test_error += loss_function(o, digit)

            confidence, prediction = find_winner(o)

            if prediction == digit:
                correct += 1

            print(f"Result: {prediction == digit}, Confidence: {confidence * 100}%, Output: {o}")

        output_file.write(f"{epoch}, {global_error / len(x_train)}, {global_test_error / len(x_test)}\n")
        output_file.flush()

        print(f"Finished Epoch: {epoch}, Global Training Error: {global_error / len(x_train)}")
        print(f"Test Correct: {correct / len(x_test) * 100}%, Global Test Error: {global_test_error / len(x_test)}")
        


    output_file.close()


# Mean Squared error accross output vector
def calculate_error_MSE(o, y):
    expected_output = [0.0 for _ in range(len(o))]
    expected_output[y] = 1.0
    
    res = (o - expected_output)**2

    return np.mean(res)

# Categorical Cross-Entropy
def calculate_error_CCE(o, y):
    expected_output = [0.0 for _ in range(len(o))]
    expected_output[y] = 1.0

    clip_val = 1e-12
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