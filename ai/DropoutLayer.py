import numpy as np


class DropoutLayer:
    """
    Dropout inverted (scale pendant le train, pas pendant l'inférence).
    training=True  → masque aléatoire + scale par 1/(1-rate)
    training=False → pass-through (inférence / validation)
    """

    def __init__(self, rate=0.5):
        self.rate = rate
        self.training = True
        self._mask = None

    def forward(self, x):
        if not self.training:
            return x
        self._mask = (np.random.rand(*x.shape) > self.rate).astype(x.dtype)
        return x * self._mask / (1.0 - self.rate)

    def backward(self, grad):
        return grad * self._mask / (1.0 - self.rate)
