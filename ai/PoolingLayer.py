from xp import xp as np


class PoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self._last_ndim = None
        # Format 4D legacy [batch, C, H, W] — inférence (main.py)
        self._switches_4d     = None
        self._input_shape_4d  = None
        # Format 3D [H, W, D] — entraînement image seule
        self._switches_3d_h   = None
        self._switches_3d_w   = None
        self._input_shape_3d  = None

    def forward(self, input_data):
        if input_data.ndim == 3:
            self._last_ndim = 3
            return self._forward_3d(input_data)
        # ndim == 4 : distinguer [batch, H, W, D] (channels-last) de [batch, C, H, W]
        # Convention : le dernier axe est les canaux pour le pipeline d'entraînement
        if input_data.ndim == 4:
            # Le pipeline d'entraînement produit toujours [batch, H, W, D]
            # → traité comme batched channels-last si _last_ndim est 3 ou batch
            # On différencie par la signature appelante : main_train passe 4D CL,
            # main.py passe 4D CF. On utilise la convention "batch>1 et ndim=4"
            # pour distinguer, mais la seule façon fiable est le flag ci-dessous.
            if getattr(self, '_batch_mode', False):
                self._last_ndim = 'batch'
                return self._forward_batch(input_data)
        self._last_ndim = 4
        return self._forward_4d(input_data)

    def forward_batch(self, input_data):
        """Entrée [batch, H, W, D] channels-last — utilisé par main_train."""
        self._last_ndim = 'batch'
        return self._forward_batch(input_data)

    # ── Format 4D legacy [batch, C, H, W] — inférence ─────────────────────────

    def _forward_4d(self, input_data):
        batch_size, channels, height, width = input_data.shape
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))
        self._switches_4d = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        self._input_shape_4d = input_data.shape

        for i in range(out_h):
            for j in range(out_w):
                h_s = i * self.stride
                w_s = j * self.stride
                region = input_data[:, :, h_s:h_s + self.pool_size, w_s:w_s + self.pool_size]
                output[:, :, i, j] = np.max(region, axis=(2, 3))
                flat = region.reshape(batch_size, channels, -1)
                idx = np.argmax(flat, axis=2)
                self._switches_4d[:, :, i, j, 0] = idx // self.pool_size + h_s
                self._switches_4d[:, :, i, j, 1] = idx % self.pool_size + w_s

        return output

    # ── Format 3D [H, W, D] — image seule ─────────────────────────────────────

    def _forward_3d(self, input_data):
        H, W, D = input_data.shape
        p, s = self.pool_size, self.stride
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1
        self._input_shape_3d = input_data.shape

        sh, sw, sd = input_data.strides
        windows = np.lib.stride_tricks.as_strided(
            input_data,
            shape=(out_h, out_w, p, p, D),
            strides=(sh * s, sw * s, sh, sw, sd),
        )
        windows_flat = windows.reshape(out_h, out_w, p * p, D)

        idx_flat = np.argmax(windows_flat, axis=2)
        output   = np.max(windows_flat, axis=2)

        row_off = (idx_flat // p).astype(np.intp)
        col_off = (idx_flat % p).astype(np.intp)

        oi = np.arange(out_h, dtype=np.intp)[:, None, None]
        oj = np.arange(out_w, dtype=np.intp)[None, :, None]

        self._switches_3d_h = row_off + oi * s
        self._switches_3d_w = col_off + oj * s

        return output

    # ── Format 4D batché [batch, H, W, D] — entraînement ──────────────────────

    def _forward_batch(self, input_data):
        """[batch, H, W, D] → [batch, out_h, out_w, D]"""
        batch, H, W, D = input_data.shape
        p, s = self.pool_size, self.stride
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1
        self._input_shape_3d = input_data.shape   # stocké pour backward

        sb, sh, sw, sd = input_data.strides
        windows = np.lib.stride_tricks.as_strided(
            input_data,
            shape=(batch, out_h, out_w, p, p, D),
            strides=(sb, sh * s, sw * s, sh, sw, sd),
        )
        windows_flat = windows.reshape(batch, out_h, out_w, p * p, D)

        idx_flat = np.argmax(windows_flat, axis=3)   # [batch, out_h, out_w, D]
        output   = np.max(windows_flat, axis=3)      # [batch, out_h, out_w, D]

        row_off = (idx_flat // p).astype(np.intp)
        col_off = (idx_flat % p).astype(np.intp)

        oi = np.arange(out_h, dtype=np.intp)[None, :, None, None]   # [1, out_h, 1, 1]
        oj = np.arange(out_w, dtype=np.intp)[None, None, :, None]   # [1, 1, out_w, 1]

        self._switches_3d_h = row_off + oi * s   # [batch, out_h, out_w, D]
        self._switches_3d_w = col_off + oj * s

        return output

    # ── Backward ───────────────────────────────────────────────────────────────

    def backward(self, grad):
        if self._last_ndim == 3:
            return self._backward_3d(grad)
        if self._last_ndim == 'batch':
            return self._backward_batch(grad)
        return self._backward_4d(grad)

    def _backward_4d(self, grad):
        batch_size, channels, H, W = self._input_shape_4d
        d_input = np.zeros(self._input_shape_4d, dtype=grad.dtype)
        out_h, out_w = grad.shape[2], grad.shape[3]

        for i in range(out_h):
            for j in range(out_w):
                for b in range(batch_size):
                    for c in range(channels):
                        h = self._switches_4d[b, c, i, j, 0]
                        w = self._switches_4d[b, c, i, j, 1]
                        d_input[b, c, h, w] += grad[b, c, i, j]

        return d_input

    def _backward_3d(self, grad):
        """Scatter vectorisé pour image seule [H, W, D]."""
        d_input = np.zeros(self._input_shape_3d, dtype=grad.dtype)
        out_h, out_w, D = grad.shape

        d_ch = np.broadcast_to(
            np.arange(D, dtype=np.intp)[None, None, :],
            (out_h, out_w, D),
        )
        np.add.at(d_input, (self._switches_3d_h, self._switches_3d_w, d_ch), grad)
        return d_input

    def _backward_batch(self, grad):
        """
        Scatter vectorisé pour mini-batch [batch, H, W, D].
        stride == pool_size (non-overlapping) → assignment directe, pas d'atomique.
        """
        batch, H, W, D = self._input_shape_3d
        d_input = np.zeros(self._input_shape_3d, dtype=grad.dtype)
        out_h, out_w = grad.shape[1], grad.shape[2]

        b_idx = np.arange(batch, dtype=np.intp)[:, None, None, None]          # [batch,1,1,1]
        d_ch  = np.arange(D,     dtype=np.intp)[None, None, None, :]           # [1,1,1,D]

        # Pour stride >= pool_size (fenêtres non-chevauchantes), chaque position
        # est ciblée au plus une fois → assignment directe sans atomique.
        d_input[b_idx, self._switches_3d_h, self._switches_3d_w, d_ch] = grad
        return d_input
