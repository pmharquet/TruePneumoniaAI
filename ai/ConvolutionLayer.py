from xp import xp as np


class ConvolutionLayer:
    def __init__(self, kernel, stride):
        self.kernel = np.array(kernel, dtype=np.float32)  # [nb_filtres, kH, kW, D_entrée]
        self.stride = stride
        self._last_input = None
        self.d_kernel = np.zeros_like(self.kernel)

    @classmethod
    def create(cls, nb_filtres, kH, kW, D_entree, stride=1):
        """Initialise un ConvolutionLayer avec He initialization."""
        scale = np.sqrt(2.0 / (kH * kW * D_entree))
        kernel = (np.random.randn(nb_filtres, kH, kW, D_entree) * scale).astype(np.float32)
        return cls(kernel, stride)

    def patch_generator(self, image):
        nb_filtres, kernel_height, kernel_width, kernel_channels = self.kernel.shape
        image_height, image_width, image_channels = image.shape
        for i in range(0, image_height - kernel_height + 1, self.stride):
            for j in range(0, image_width - kernel_width + 1, self.stride):
                yield i, j, image[i:i+kernel_height, j:j+kernel_width, :]

    def kernel_convolution(self, patch, filtre):
        return np.sum(patch * filtre)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, image):
        """
        Accepte :
          - [H, W]          → image grayscale single
          - [H, W, D]       → image single (channels-last)
          - [batch, H, W, D]→ mini-batch (channels-last)
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        self._last_input = image

        if image.ndim == 4:
            return self._forward_batched(image)
        return self._forward_single(image)

    def _forward_single(self, image):
        """[H, W, D] → [out_H, out_W, nb_filtres]"""
        nb_filtres, kH, kW, kC = self.kernel.shape
        img_H, img_W, _ = image.shape
        out_H = (img_H - kH) // self.stride + 1
        out_W = (img_W - kW) // self.stride + 1

        s0, s1, s2 = image.strides
        patches = np.lib.stride_tricks.as_strided(
            image,
            shape=(out_H, out_W, kH, kW, kC),
            strides=(s0 * self.stride, s1 * self.stride, s0, s1, s2),
        )
        patches_2d = patches.reshape(out_H * out_W, kH * kW * kC)
        kernel_2d  = self.kernel.reshape(nb_filtres, kH * kW * kC)
        return (patches_2d @ kernel_2d.T).reshape(out_H, out_W, nb_filtres)

    def _forward_batched(self, image):
        """[batch, H, W, D] → [batch, out_H, out_W, nb_filtres]"""
        nb_filtres, kH, kW, kC = self.kernel.shape
        batch, img_H, img_W, _ = image.shape
        out_H = (img_H - kH) // self.stride + 1
        out_W = (img_W - kW) // self.stride + 1

        sb, s0, s1, s2 = image.strides
        patches = np.lib.stride_tricks.as_strided(
            image,
            shape=(batch, out_H, out_W, kH, kW, kC),
            strides=(sb, s0 * self.stride, s1 * self.stride, s0, s1, s2),
        )
        patches_2d = patches.reshape(batch * out_H * out_W, kH * kW * kC)
        kernel_2d  = self.kernel.reshape(nb_filtres, kH * kW * kC)
        return (patches_2d @ kernel_2d.T).reshape(batch, out_H, out_W, nb_filtres)

    # ── Backward ───────────────────────────────────────────────────────────────

    def backward(self, grad):
        if self._last_input.ndim == 4:
            return self._backward_batched(grad)
        return self._backward_single(grad)

    def _backward_single(self, grad):
        """
        grad : [out_H, out_W, nb_filtres]
        Retourne dL/dinput : [img_H, img_W, D_entrée]
        """
        nb_filtres, kH, kW, kC = self.kernel.shape
        out_H, out_W = grad.shape[0], grad.shape[1]
        s = self.stride

        self.d_kernel = np.zeros_like(self.kernel)
        d_input = np.zeros_like(self._last_input)

        grad_2d = grad.reshape(out_H * out_W, nb_filtres)

        for kh in range(kH):
            for kw in range(kW):
                patch    = self._last_input[kh:kh + out_H * s:s, kw:kw + out_W * s:s, :]
                patch_2d = patch.reshape(out_H * out_W, kC)
                self.d_kernel[:, kh, kw, :] = grad_2d.T @ patch_2d
                d_input[kh:kh + out_H * s:s, kw:kw + out_W * s:s, :] += (
                    grad_2d @ self.kernel[:, kh, kw, :]
                ).reshape(out_H, out_W, kC)

        return d_input

    def _backward_batched(self, grad):
        """
        grad : [batch, out_H, out_W, nb_filtres]
        d_kernel est la somme des gradients sur le batch (on divisera par batch_size dans main_train).
        """
        nb_filtres, kH, kW, kC = self.kernel.shape
        batch, out_H, out_W, _ = grad.shape
        s = self.stride

        self.d_kernel = np.zeros_like(self.kernel)
        d_input = np.zeros_like(self._last_input)

        # grad_2d : [batch * out_H * out_W, nb_filtres]
        grad_2d = grad.reshape(batch * out_H * out_W, nb_filtres)

        for kh in range(kH):
            for kw in range(kW):
                patch    = self._last_input[:, kh:kh + out_H * s:s, kw:kw + out_W * s:s, :]
                patch_2d = patch.reshape(batch * out_H * out_W, kC)
                self.d_kernel[:, kh, kw, :] = grad_2d.T @ patch_2d
                d_input[:, kh:kh + out_H * s:s, kw:kw + out_W * s:s, :] += (
                    grad_2d @ self.kernel[:, kh, kw, :]
                ).reshape(batch, out_H, out_W, kC)

        return d_input

    def zero_grads(self):
        self.d_kernel[:] = 0.0

    def get_params_and_grads(self):
        return [(self.kernel, self.d_kernel)]
