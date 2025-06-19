import numpy as np

##

MNIST_DATASET_TRAIN = np.loadtxt('src/mnist_csv/mnist_train.csv', delimiter=',', skiprows=1)
MNIST_DATASET_TEST = np.loadtxt('src/mnist_csv/mnist_test.csv', delimiter=',', skiprows=1)

TRAIN_LABELS = MNIST_DATASET_TRAIN[:, 0]
TRAIN_FEATURES = MNIST_DATASET_TRAIN[:, 1:]

TEST_LABELS = MNIST_DATASET_TEST[:, 0]
TEST_FEATURES = MNIST_DATASET_TEST[:, 1:]

TRAIN_FEATURES = TRAIN_FEATURES.reshape(60000, 28, 28)
TEST_FEATURES = TEST_FEATURES.reshape(10000, 28, 28)

##

PARTITIONED_TRAIN_FEATURES = TRAIN_FEATURES.reshape(600, 100, 28, 28)
PARTITIONED_TEST_FEATURES = TEST_FEATURES.reshape(100, 100, 28, 28)

##

import mlp