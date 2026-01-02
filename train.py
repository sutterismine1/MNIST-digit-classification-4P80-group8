# COSC 4P80 Project Option 1: CNN from scrath to classify Mnist digits
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
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

# Used for keeping track of average training error in block_error output file
BLOCK_SIZE = 250

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
    assert 'seed' in config, "The config must provide a value for \'seed\'"
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    assert 'epochs' in config, "The config must provide a value for \'epochs\'"
    epochs = config['epochs']

    assert 'output_filename' in config, "The config must provide a value for \'output_filename\'"
    output_filename = config['output_filename']
    block_error_output_filename = config['output_filename'] + "_block_error"

    print("Reading Data")
    (x_train, y_train), (x_test, y_test) = load_MNIST_data(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    
    x_train = np.array(x_train, dtype=float) / 255.0
    x_test  = np.array(x_test, dtype=float) / 255.0

    # Create an ordering that will be shuffled between epochs
    data_map = [x for x in range(len(x_train))]

    loss_function = calculate_error_CCE         # CHANGE LOSS FUNCTION HERE AND IN learn_output() OF FullyConnectedLayer.py

    print("Initializing Network")
    network = ConvolutionalNetwork(config)

    output_file = open(output_filename+".csv", "w")
    output_file.write("Epoch, Training Set, Test Set, Test Correct\n")
    output_file.flush()

    block_error_output_file = open(block_error_output_filename+".csv", "w")
    block_error_output_file.write("Trained Samples, Average Training Error\n")
    block_error_output_file.flush()

    print("Starting Training")
    step = 0    # used to keep track of how many samples have been trained on
    block_error = 0
    for epoch in range(1, epochs+1):
        global_error = 0

        for index in data_map:
            o = network.apply_and_learn(x_train[index], y_train[index])

            error = loss_function(o, y_train[index])
            global_error += error
            block_error += error
            step += 1

            if step % BLOCK_SIZE == 0 and step != 0:
                block_error_output_file.write(f"{step}, {block_error / BLOCK_SIZE}\n")
                block_error_output_file.flush()
                block_error = 0
                print(step)

        # shuffle data_map
        for i in range(math.floor(len(x_train) * 0.20)):
            r1, r2 = random.randint(0, len(x_train)-1), random.randint(0, len(x_train)-1)
            data_map[r1], data_map[r2] = data_map[r2], data_map[r1]

        print("Running Against Test Set")
        global_test_error = 0
        correct = 0
        for i in range(len(x_test)):
            image, digit = x_test[i], y_test[i]   # Just testing on the training set until the CNN is correct
            o = network.apply(image)

            global_test_error += loss_function(o, digit)

            confidence, prediction = find_winner(o)

            if prediction == digit:
                correct += 1

            #print(f"Result: {prediction == digit}, Confidence: {confidence * 100}%, Output: {o}")

        output_file.write(f"{epoch}, {global_error / len(x_train)}, {global_test_error / len(x_test)}, {correct / len(x_test) * 100}%\n")
        output_file.flush()

        network.increase_epoch()
        save_network(network.get_network_variables(config))

        print(f"Finished Epoch: {epoch}, Global Training Error: {global_error / len(x_train)}")
        print(f"Test Correct: {correct / len(x_test) * 100}%, Global Test Error: {global_test_error / len(x_test)}")

    output_file.close()
    block_error_output_file.close()

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

# write network config to file
def save_network(config):
        name = config["output_filename"]
        if "trained_epochs" in config:
            name += f"_{config["trained_epochs"]}"
        name += ".json"
        with open(name, "w") as file:
            json.dump(config, file, indent=4)


if __name__ == "__main__":
    main()