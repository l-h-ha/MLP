import numpy as np
from mlp.base_object import base_object

class MEAN_SQUARED_ERROR(base_object):
    def __init__(self):
        self.obj_type = "loss_derivative"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return (2 / y_true.size) * (y_pred - y_true)