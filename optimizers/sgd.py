from .base import Optimizer


class SGD(Optimizer):
    """x <- x - lr * g"""

    def __init__(self, lr=0.01):
        super().__init__(lr)

    def step(self, x, grad, grad_fn=None):
        return x - self.lr * grad
