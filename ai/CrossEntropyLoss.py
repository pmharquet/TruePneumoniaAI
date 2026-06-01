import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self._predictions = None
        self._label = None

    def forward(self, predictions, label):
        self._predictions = predictions
        self._label = label
        return -np.log(predictions[label] + 1e-10)

    def backward(self):
        grad = self._predictions.copy()
        grad[self._label] -= 1.0
        return grad
