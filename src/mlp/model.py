from mlp.losses import losses, loss_derivatives
from mlp import layer
import numpy as np

class model():
    def __init__(self, loss_str: str = "mean_squared_error", learning_rate: float = 0.1):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_function = losses[loss_str]
        self.loss_function_derivative = loss_derivatives[loss_str]

    def set_layers(self, layers_list: list[layer]) -> None:
        self.layers = layers_list

        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            current_layer.a = np.zeros(current_layer.length).reshape(current_layer.length, 1)
            current_layer.z = np.zeros(current_layer.length).reshape(current_layer.length, 1)

            next_layer.a = np.zeros(next_layer.length).reshape(next_layer.length, 1)
            next_layer.z = np.zeros(next_layer.length).reshape(next_layer.length, 1)

            current_layer.w = np.random.randn(next_layer.length, current_layer.length) - 0.5
            current_layer.b = np.random.randn(next_layer.length, 1) - 0.5

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layers[0].a = inputs

        for i in range(len(self.layers) - 1):
            this_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            next_layer.z = np.dot(this_layer.w, this_layer.a) + this_layer.b
            next_layer.a = next_layer.activation(next_layer.z, next_layer.activation_hyperparameters)

        return self.layers[-1].a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        gradients = {"dW" : [None] * (len(self.layers) - 1), "dB" : [None] * (len(self.layers) - 1)}
        dC_wrt_a = self.loss_function_derivative(y_pred, y_true) # (z, 1)
        loss = self.loss_function(y_pred, y_true)

        for i in reversed(range(1, len(self.layers))):
            this_layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            dZ_last = this_layer.derivative(this_layer.z, this_layer.derivative_hyperparameters) # (z, 1)
            dC_wrt_z = dC_wrt_a * dZ_last # (z, 1)

            # W_prev = (z, y) => dC/dW = (z, 1) @ (1, y)
            dC_wrt_w = np.dot(dC_wrt_z, prev_layer.a.T)
            dC_wrt_b = dC_wrt_z * 1 # (z, 1)

            # dC/dA_prev = (y, 1) = (y, 1) * (z, 1)
            dC_wrt_a = np.dot(prev_layer.w.T, dC_wrt_z)

            gradients["dW"][i - 1] = dC_wrt_w
            gradients["dB"][i - 1] = dC_wrt_b

        for i in range(len(self.layers) - 1):
            self.layers[i].w -= self.learning_rate * gradients["dW"][i]
            self.layers[i].b -= self.learning_rate * gradients["dB"][i]

        return loss