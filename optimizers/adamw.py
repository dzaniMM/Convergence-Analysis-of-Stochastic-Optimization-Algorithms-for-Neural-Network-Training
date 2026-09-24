import numpy as np

from .base import Optimizer


class AdamW(Optimizer):
    """Adam sa razdvojenim (decoupled) weight decay-om.
    m <- b1*m + (1-b1)*g;  v <- b2*v + (1-b2)*g^2
    m_hat = m/(1-b1^t);  v_hat = v/(1-b2^t)
    x <- x - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * x)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.reset()

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def step(self, x, grad, grad_fn=None):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * x)
