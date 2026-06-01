import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights, dtype=np.float64)
        self.bias = np.array([float(bias)], dtype=np.float64)
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros(1, dtype=np.float64)
        self._last_input = None

    def forward(self, input_data):
        self._last_input = input_data
        return np.sum(input_data * self.weights) + self.bias[0]

    def backward(self, grad_scalar):
        # Accumulation : permet d'appeler backward plusieurs fois par batch
        # avant zero_grads (appelé une seule fois en début de batch).
        self.d_weights += grad_scalar * self._last_input
        self.d_bias[0] += grad_scalar

    def get_params_and_grads(self):
        return [(self.weights, self.d_weights), (self.bias, self.d_bias)]
