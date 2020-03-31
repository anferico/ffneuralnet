# Datasets
The [Monk](Monk) directory contains train and test splits of the MONK's problems datasets. 
The datasets have already been one-hot encoded. 
For information about the dataset itself, please refer to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems).

The [ML_Cup](ML_Cup) directory, instead, contains a dataset of sensor readings.
The task is to predict the last 2 columns using the first 10 as features. 
The dataset is divided into a training set and a blind test set, meaning that target values aren't available for the latter.
Nonetheless, you can evaluate the network's performances by taking a subset of the training set and use it as a validation or test set.
