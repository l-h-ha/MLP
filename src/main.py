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

BATCH_SIZE = 30.0

if not (60000.0 / BATCH_SIZE).is_integer() and not (10000.0 / BATCH_SIZE).is_integer():
    raise ValueError("Incorrect batch size.")

PARTITIONED_TRAIN_FEATURES = TRAIN_FEATURES.reshape(60000.0 / BATCH_SIZE, BATCH_SIZE, 28, 28)
PARTITIONED_TEST_FEATURES = TEST_FEATURES.reshape(10000.0 / BATCH_SIZE, BATCH_SIZE, 28, 28)

##

from mlp import model, layer
model = model(learning_rate=0.1, loss_str="categorial_cross_entropy")
model.set_layers([
    layer(784, "leaky_relu", {"a": 0.01}, {"a": 0.01}),
    layer(10, "sigmoid"),
    layer(10, "softmax")
])

for i in range(100):
    feed = TRAIN_FEATURES[i].flatten().reshape(28*28, 1)
    feed = 2 * (feed - np.min(feed)) / (np.max(feed) - np.min(feed)) - 1
    output = model.forward(feed)

    y_true = [0] * 10
    y_true[TRAIN_LABELS[i].astype(int) - 1] = 1
    y_true = np.array(y_true).reshape(10, 1)
    loss = model.backward(y_true, output)
    print("Loss:", loss)