from xp import xp as np


class GlobalAveragePoolingLayer:
    def __init__(self):
        self._input_shape = None

    def forward(self, input_data):
        self._input_shape = input_data.shape
        if input_data.ndim == 4:
            # Batché [batch, H, W, D] → [batch, D]
            return np.mean(input_data, axis=(1, 2))
        if input_data.ndim != 3:
            raise ValueError("Input must be 3D [H,W,D] or 4D [batch,H,W,D]")
        # Single [H, W, D] → [D]
        return np.mean(input_data, axis=(0, 1))

    def backward(self, grad):
        if len(self._input_shape) == 4:
            # Batché : grad [batch, D] → [batch, H, W, D]
            batch, H, W, D = self._input_shape
            return np.ones(self._input_shape, dtype=grad.dtype) * grad[:, np.newaxis, np.newaxis, :] / (H * W)
        # Single : grad [D] → [H, W, D]
        H, W, D = self._input_shape
        return np.ones(self._input_shape, dtype=grad.dtype) * grad[np.newaxis, np.newaxis, :] / (H * W)
