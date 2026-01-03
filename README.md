# MNIST Digit Classification

A group project where we implemented a CNN from scratch in Python using numpy in order to classify the MNIST digit dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

# Group
| Name               | Github Link                        |
|--------------------|------------------------------------|
| Nicholas Parise    | https://github.com/Nicholas-Parise |
| Geoffrey Jensen    | https://github.com/Gexff           |
| Stephen Stefanidis | https://github.com/sutterismine1   |

# Sample Executions
Training a model:
```
python train.py configs/config_A.json
```

Running a trained model against the test set:
```
python test.py models/CNN_A.json
```

Draw your own digits to run against a trained model:
```
python draw.py models/CNN_C.json
```

# Model Config JSON Format
Training file definition:
```
{
    "seed": INTEGER,                // The seed in which the random number generator will be initialized with. Random numbers are used to set initial model weights, and to shuffle the training set after every epoch.
    "learning_rate": FLOAT,         // The static learning rate to train the network with. Typically between 0.01-0.0001.
    "epochs": INTEGER,              // The maximim number of epochs to run for.
    "output_filename": STRING,      // The filename used for the output files. Two CSVs are created, one with statistics for each completed epoch, and the other with the training error by block.
    "convolutional_layers": [       // Array of definitions for convolutional layers. There must be at least 1.
        {                                // Definition for a convolutional layer
            "kernel_count": INTEGER,     // The number of kernels in the layer
            "kernel_dim": INTEGER,       // The dimensions of the kernels in this layer. Defines both x and y dimensions.
            "padding": INTEGER,          // The amount of padding before the kernel computations. A value of 1 will pad a 0 on both sides of both x and y axes.
            "pooling_dim": INTEGER       // The dimension of the pooling filter. The stride will be equal to the dimension. Defines both x and y dimensions of the pooling filter.
        }
    ],
    "fully_connected_layers": [     // Array of definitions for fully connected sigmoid layers. May be omitted or empty, inwhich case there is only the output softmax fully connected layer.
        {                                // Definiton for a fully connected sigmoid layer.
            "node_count": INTEGER        // The number of neurons/nodes in the fully connected sigmoid layer.
        }
    ]
}
```

Trained models are saved like a config file, but include the weights and biases for every defined layer, plus the output layer.

Trained model's will also have an additional name/value pair of "trained_epochs": INTEGER, which is the number of epochs the model was trained for. Because of the way the model is saved, this name/value pair will be mixed in the middle of the file somewhere, so use CTRL+F to find it easily.
