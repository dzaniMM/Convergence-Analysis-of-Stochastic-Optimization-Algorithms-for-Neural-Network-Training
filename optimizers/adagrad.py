import numpy as np

from .base import Optimizer


class AdaGrad(Optimizer):
    """G <- G + g^2;  x <- x - lr * g / (sqrt(G) + eps)"""

    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = None

    def reset(self):
        self.G = None

    def step(self, x, grad, grad_fn=None):
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G += grad**2
        return x - self.lr * grad / (np.sqrt(self.G) + self.eps)
