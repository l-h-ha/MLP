from mlp import layer
from mlp.base_object import base_object

import numpy as np

class model():
    def __init__(self, learning_rate: float = 0.1):
        self.architecture: list[base_object] = []
        self.layers: list[layer] = []
        self.activations: list[base_object] = []
        self.derivatives: list[base_object] = []
        self.weight_modifiers: list[base_object] = [None]

        self.learning_rate: base_object = learning_rate
        self.loss_function: base_object = None
        self.loss_function_derivative = None

    def set_architecture(self, architecture: list[base_object]) -> None:
        self.architecture = architecture

        for component in architecture:
            if component.get_type() == "layer":
                self.layers.append(component)
            elif component.get_type() == "activation":
                self.activations.append(component)
            elif component.get_type() == "derivative":
                self.derivatives.append(component)
            elif component.get_type() == "weight_modifier":
                self.weight_modifiers.append(component)
            elif component.get_type() == "loss":
                self.loss_function = component
            elif component.get_type() == "loss_derivative":
                self.loss_function_derivative = component

        for i in range(len(self.layers)):
            this_layer = self.layers[i]
            this_layer.a = np.zeros(this_layer.shape)
            
            if i >= 1:
                prev_layer = self.layers[i - 1]
                this_layer.z = np.zeros(this_layer.shape)
                this_layer.b = np.zeros(this_layer.shape)

                if self.weight_modifiers[i] is not None:
                    this_layer.w = self.weight_modifiers[i].init(prev_layer.shape, this_layer.shape)
                else:
                    this_layer.w = np.random.rand(prev_layer.shape[1], this_layer.shape[1])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layers[0].a = self.activations[0](inputs)

        for i in range(len(self.layers) - 1):
            this_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            next_layer.z = np.dot(this_layer.a, next_layer.w) + next_layer.b
            next_layer.a = self.activations[i + 1](next_layer.z)

        return self.layers[-1].a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        dC_wrt_a = self.loss_function_derivative(y_pred, y_true) # (m, z)
        loss = self.loss_function(y_pred, y_true) # scalar
        batch_size_inverse = 1 / dC_wrt_a.shape[0] # m

        print(f"LOSS: {loss}")

        for i in reversed(range(1, len(self.layers))):
            this_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            
            dA_wrt_z = self.derivatives[i](this_layer.z) # (m, z)
            dC_wrt_z = dC_wrt_a * dA_wrt_z # (m, z)

            # dC/dW = (dZ/dW).T * dC/dZ
            #       =   A.T  *  ...
            # (y, z) = (m, y).T @ (m, z)

            dC_wrt_w = batch_size_inverse * np.dot(prev_layer.a.T, dC_wrt_z)
            dC_wrt_b = batch_size_inverse * np.sum(dC_wrt_z, axis=0, keepdims=True) # (m, z) -> (1, z)

            this_layer.w -= self.learning_rate * dC_wrt_w
            this_layer.b -= self.learning_rate * dC_wrt_b

            # dC/dA_prev = dC/dZ * dZ/dA_prev
            # (m, y) = (m, z) * (y, z).T

            dC_wrt_a = np.dot(dC_wrt_z, this_layer.w.T)

        return loss