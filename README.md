# FeedForward Neural Network
Implementation of a FeedForward Neural Network (FFNN) using Matlab. The network is trained using Gradient Descent combined with the BackPropagation technique, and its performances can be evaluated on two datasets, namely the Monk dataset (https://archive.ics.uci.edu/ml/datasets/MONK's+Problems) and a dataset of sensor readings.

The file [neural_network.m](neural_network.m) carries out the whole training process (also plotting the learning curves and saving them to a file) given a set of hyperparameters. If you want to perform a grid search to do hyperparametrs tuning, run [main.m](main.m) instead.

The [Datasets](Datasets) directory contains a couple of dataset that can be used to train the network and evaluate its performances.
