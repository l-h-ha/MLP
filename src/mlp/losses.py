import numpy as np

def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray):
    error_sq = (y_pred - y_true)**2
    return np.mean(error_sq)

def mean_squared_error_derivative(y_pred: np.ndarray, y_true: np.ndarray):
    return (2 / y_pred.shape[0]) * (y_pred - y_true)

def categorial_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray):
    epsilon = 1e-15 # prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred))

def categorial_cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray):
    return y_pred - y_true

##########

losses = {
    "mean_squared_error": mean_squared_error,
    "categorial_cross_entropy": categorial_cross_entropy
}

loss_derivatives = {
    "mean_squared_error": mean_squared_error_derivative,
    "categorial_cross_entropy": categorial_cross_entropy_derivative
}