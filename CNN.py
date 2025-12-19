from dataclasses import dataclass
# dataclass used for configurating hyperparameters and storing default values
@dataclass
class CNNHyperParams: # data classes are basically mutable named tuples with default values
    stride: int = 1
    num_filters_conv1: int = 6
    num_filters_conv2: int = 16
    filter_size: int = 5

# The main CNN class
# Only works with single channel images
class CNN:
    def __init__(self, num_classes, **hyperparams):
        self.num_classes = num_classes
        self.hp = CNNHyperParams(**hyperparams)
        print("Initialized a CNN with the following parameters: ")
        print(f"Num Classes: {num_classes}")
        for attr_name, attr_value in self.hp.__dict__.items():   # iterates through a classes attributes
            print(f"{attr_name}: {attr_value}")
    def train(self):
        pass
    def test_specific_matrix(self, matrix):
        pass