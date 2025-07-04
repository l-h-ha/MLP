import numpy as np
from mlp.base_object import base_object

class mean_squared_error(base_object):
    def __init__(self):
        self.obj_type = "loss"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.square(y_pred - y_true))