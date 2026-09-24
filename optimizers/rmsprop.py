import numpy as np

from .base import Optimizer


class RMSProp(Optimizer):
    """E <- rho*E + (1-rho)*g^2;  x <- x - lr * g / (sqrt(E) + eps)"""

    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.E = None

    def reset(self):
        self.E = None

    def step(self, x, grad, grad_fn=None):
        if self.E is None:
            self.E = np.zeros_like(x)
        self.E = self.rho * self.E + (1 - self.rho) * grad**2
        return x - self.lr * grad / (np.sqrt(self.E) + self.eps)
