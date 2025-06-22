from mlp.activations import activations, derivatives

class layer():
    def __init__(self, length: int, activation: str, activation_hyperparameters=None, derivative_hyperparameters=None):
        self.activation = activations[activation]
        self.derivative = derivatives[activation]

        self.w = None
        self.b = None

        self.z = None
        self.a = None

        self.length = length
        
        self.activation_hyperparameters = activation_hyperparameters
        self.derivative_hyperparameters = derivative_hyperparameters