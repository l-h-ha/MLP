import numpy as np

##

MNIST_DATASET_TRAIN = np.loadtxt('src/mnist_csv/mnist_train.csv', delimiter=',', skiprows=1)
MNIST_DATASET_TEST = np.loadtxt('src/mnist_csv/mnist_test.csv', delimiter=',', skiprows=1)

TRAIN_LABELS = MNIST_DATASET_TRAIN[:, 0]
TRAIN_FEATURES = MNIST_DATASET_TRAIN[:, 1:]

TEST_LABELS = MNIST_DATASET_TEST[:, 0]
TEST_FEATURES = MNIST_DATASET_TEST[:, 1:]

TRAIN_FEATURES = (TRAIN_FEATURES.reshape(60000, 784).astype('float32') - (255.0 / 2)) / (255.0 / 2)
TEST_FEATURES = (TEST_FEATURES.reshape(10000, 784).astype('float32') - (255.0 / 2)) / (255.0 / 2)

##

BATCH_SIZE = 100

PARTITIONED_TRAIN_FEATURES = TRAIN_FEATURES.reshape(60000 // BATCH_SIZE, BATCH_SIZE, 784)
PARTITIONED_TEST_FEATURES = TEST_FEATURES.reshape(10000 // BATCH_SIZE, BATCH_SIZE, 784)

PARTITIONED_TRAIN_LABELS = TRAIN_LABELS.reshape(60000 // BATCH_SIZE, BATCH_SIZE, 1)
PARTITIONED_TEST_LABELS = TEST_LABELS.reshape(10000 // BATCH_SIZE, BATCH_SIZE, 1)

'''
X = (BATCH_SIZE, 28**2)
Y = (BATCH_SIZE, 10)

Y_true = (BATCH_SIZE, 10)
Y_label = (BATCH_SIZE, 1)
'''

##

from mlp import model, layer
from mlp.activations import leaky_relu, softmax, identity
from mlp.derivatives import LEAKY_RELU, SOFTMAX, IDENTITY
from mlp.weight_modifiers import HE_init
from mlp.losses import mean_squared_error
from mlp.loss_derivatives import MEAN_SQUARED_ERROR

model = model(learning_rate=0.1)
model.set_architecture([
    identity(), IDENTITY(),
    layer(shape=(BATCH_SIZE, 784)),

    leaky_relu(), LEAKY_RELU(), HE_init(),
    layer(shape=(BATCH_SIZE, 10)),

    softmax(), SOFTMAX(), HE_init(),
    layer(shape=(BATCH_SIZE, 10)),

    mean_squared_error(), MEAN_SQUARED_ERROR()
])

for i in range(1000):
    feed = PARTITIONED_TRAIN_FEATURES[0]
    output = model.forward(feed)

    
    ##

    y_true = np.zeros(shape=(BATCH_SIZE, 10))
    y_label = PARTITIONED_TRAIN_LABELS[0].flatten().astype('int')
    rows = np.arange(BATCH_SIZE)
    y_true[rows, y_label] = 1 

    ##

    loss = model.backward(y_true, output)
    #print('Loss:', loss)
    