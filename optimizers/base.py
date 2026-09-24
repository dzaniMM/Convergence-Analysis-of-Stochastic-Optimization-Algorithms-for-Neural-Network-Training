import numpy as np


class Optimizer:
    """Osnovna klasa: step(x, grad) vraća novu tačku, stanje (brzina, akumulatori) čuva optimizator."""

    def __init__(self, lr):
        self.lr = lr

    def step(self, x, grad, grad_fn=None):
        """grad je gradijent u x. grad_fn(x) je opcioni pozivalac gradijenta u proizvoljnoj
        tački (treba samo Nesterovu, koji računa gradijent u look-ahead tački)."""
        raise NotImplementedError

    def reset(self):
        """Briše interno stanje da bi se optimizator mogao ponovo koristiti."""

    def minimize(self, f, grad_f, x0, n_iters=1000, tol=0.0):
        """Pokreće n_iters koraka od x0. Vraća putanju (n+1, dim) i vrednosti f duž nje."""
        self.reset()
        x = np.asarray(x0, dtype=float).copy()
        path, values = [x.copy()], [f(x)]
        for _ in range(n_iters):
            g = grad_f(x)
            if tol and np.linalg.norm(g) < tol:
                break
            x = self.step(x, g, grad_f)
            path.append(x.copy())
            values.append(f(x))
        return np.array(path), np.array(values)
