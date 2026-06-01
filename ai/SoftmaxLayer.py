import numpy as np


class SoftmaxLayer:
    def __init__(self):
        self._output = None

    def forward(self, input_data):
        e = np.exp(input_data - np.max(input_data))
        self._output = e / np.sum(e)
        return self._output

    def backward(self, grad):
        s = self._output
        return s * (grad - np.sum(grad * s))
