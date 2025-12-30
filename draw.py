# COSC 4P80 Project
# Geoffrey Jensen       7148710
# Stephen Stefanidis    7140030
# Nicholas Parise       7242530
#
# Draw digits and have them classified by a trained CNN model

import drawing_input
import sys
import json
from ConvolutionalNetwork import *

# Filepaths
input_path = './Data'
training_images_filepath = f'{input_path}/train-images.idx3-ubyte'
training_labels_filepath = f'{input_path}/train-labels.idx1-ubyte'
test_images_filepath = f'{input_path}/t10k-images.idx3-ubyte'
test_labels_filepath = f'{input_path}/t10k-labels.idx1-ubyte'

def main():
    # Read in arguments
    if len(sys.argv) < 2:
        print("Please provide the trained network config file.")
        exit(1)

    # Read JSON config
    config_file = sys.argv[1]
    config = None
    with open(config_file) as json_file:
        config = json.load(json_file)

    assert 'trained_epochs' in config, "The config must be a trained model!"

    print("Initializing Network")
    network = ConvolutionalNetwork(config)

    print("Opening Drawing Window")
    matrix = drawing_input.run(network)

if __name__ == "__main__":
    main()