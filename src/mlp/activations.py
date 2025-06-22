import numpy as np

def sigmoid(x: np.ndarray, AH=None):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray, AH=None):
    sx = sigmoid(x)
    return sx * (1 - sx)

def softmax(x: np.ndarray, AH=None):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def softmax_derivative(x: np.ndarray, AH=None):
    sx = softmax(x)
    return sx * (1 - sx)

def stable_softmax(x: np.ndarray, AH=None):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def stable_softmax_derivative(x: np.ndarray, AH=None):
    sx = stable_softmax(x)
    return sx * (1 - sx)

def relu(x, AH=None):
    return np.maximum(x, 0)

def relu_derivative(x, AH=None):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, AH=None):
    a = AH["a"] or 0.01
    return np.where(x > 0, x, x * a)

def leaky_relu_derivative(x, AH=None):
    a = AH["a"] or 0.01
    return np.where(x > 0, 1, a)

############

activations = {
    "relu" : relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "stable_softmax": stable_softmax
}

derivatives = {
    "relu" : relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "sigmoid": sigmoid_derivative,
    "softmax": softmax_derivative,
    "stable_softmax": stable_softmax_derivative
}