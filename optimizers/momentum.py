import numpy as np

from .base import Optimizer


class Momentum(Optimizer):
    """Klasični (heavy-ball) momentum, opciono Nesterov (NAG).

    Klasični:  v <- mu*v - lr*g(x);           x <- x + v
    Nesterov:  v <- mu*v - lr*g(x + mu*v);    x <- x + v

    Nesterov računa gradijent u look-ahead tački x + mu*v, pa mu treba grad_fn.
    Ako grad_fn nije prosleđen, koristi se gradijent u x (pa je to običan momentum).
    """

    def __init__(self, lr=0.01, momentum=0.9, nesterov=False):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.v = None

    def reset(self):
        self.v = None

    def step(self, x, grad, grad_fn=None):
        if self.v is None:
            self.v = np.zeros_like(x)
        if self.nesterov and grad_fn is not None:
            grad = grad_fn(x + self.momentum * self.v)
        self.v = self.momentum * self.v - self.lr * grad
        return x + self.v


class NAG(Momentum):
    """Nesterov Accelerated Gradient."""

    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr, momentum, nesterov=True)
