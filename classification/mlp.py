import numpy as np


class MLP:
    """Potpuno povezana mreža sa tanh ili ReLU skrivenim slojevima. Izlaz je jedan logit (binarna klasifikacija, sigmoid + BCE)
    ili k logita za k > 1 klasa (softmax + višeklasna unakrsna entropija).
    Svi parametri (W1, b1, W2, b2, ...) čuvaju se u jednom ravnom vektoru, pa se
    optimizatori (koji rade nad x ∈ R^n) mogu primeniti direktno."""

    def __init__(self, layers=(2, 16, 16, 1), activation="tanh"):
        assert activation in ("tanh", "relu")
        self.layers = tuple(layers)
        self.activation = activation
        self.shapes = list(zip(self.layers[:-1], self.layers[1:]))
        self.multiclass = self.layers[-1] > 1
        self.n_params = sum(i * o + o for i, o in self.shapes)

    def init_params(self, seed=0):
        """Glorot inicijalizacija težina za tanh (limit sqrt(6/(i+o))), He za ReLU (limit sqrt(6/i)), nulti biasi."""
        rng = np.random.default_rng(seed)
        parts = []
        for i, o in self.shapes:
            limit = np.sqrt(6.0 / i) if self.activation == "relu" else np.sqrt(6.0 / (i + o))
            parts += [rng.uniform(-limit, limit, i * o), np.zeros(o)]
        return np.concatenate(parts)

    def _unpack(self, w):
        params, k = [], 0
        for i, o in self.shapes:
            params.append((w[k:k + i * o].reshape(i, o), w[k + i * o:k + i * o + o]))
            k += i * o + o
        return params

    def _forward(self, w, X):
        params = self._unpack(w)
        acts = [X]
        for W, b in params[:-1]:
            h = acts[-1] @ W + b
            acts.append(np.maximum(h, 0) if self.activation == "relu" else np.tanh(h))
        W, b = params[-1]
        z = acts[-1] @ W + b
        return params, acts, (z if self.multiclass else z.ravel())

    def logits(self, w, X):
        return self._forward(w, X)[2]

    def predict(self, w, X):
        z = self.logits(w, X)
        return z.argmax(axis=1) if self.multiclass else (z > 0).astype(float)

    def accuracy(self, w, X, y):
        return float(np.mean(self.predict(w, X) == y))

    def loss(self, w, X, y):
        """Unakrsna entropija nad logitima (numerički stabilna): binarna, odnosno softmax za više klasa."""
        z = self.logits(w, X)
        if self.multiclass:
            z = z - z.max(axis=1, keepdims=True)
            return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))
        return float(np.mean(np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))))

    def grad(self, w, X, y):
        params, acts, z = self._forward(w, X)
        if self.multiclass:
            p = np.exp(z - z.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(y)), y] -= 1
            dz = p / len(y)
        else:
            dz = ((0.5 * (1 + np.tanh(z / 2)) - y) / len(y))[:, None]
        grads = []
        for (W, _), a in zip(reversed(params), reversed(acts)):
            grads += [dz.sum(axis=0), a.T @ dz]
            dz = (dz @ W.T) * ((a > 0) if self.activation == "relu" else (1 - a**2))
        return np.concatenate([g.ravel() for g in reversed(grads)])


class ClassificationProblem:
    """Povezuje MLP i dataset u f(w) / grad(w) nad trening skupom, u obliku koji traži Optimizer.minimize."""

    def __init__(self, dataset, layers=None, seed=0, activation="tanh"):
        """layers=None -> (broj obeležja, 16, 16, izlaz), izlaz je 1 za binarnu klasifikaciju, inače n_classes."""
        self.dataset = dataset
        if layers is None:
            layers = (dataset.X_train.shape[1], 16, 16, 1 if dataset.n_classes == 2 else dataset.n_classes)
        self.model = MLP(layers, activation)
        self.w0 = self.model.init_params(seed)

    @property
    def n_train(self):
        return len(self.dataset.y_train)

    def f(self, w):
        return self.model.loss(w, self.dataset.X_train, self.dataset.y_train)

    def grad(self, w, idx=None):
        """Gradijent nad celim trening skupom, ili nad batch-om trening primera sa indeksima idx."""
        X, y = self.dataset.X_train, self.dataset.y_train
        if idx is not None:
            X, y = X[idx], y[idx]
        return self.model.grad(w, X, y)

    def train_accuracy(self, w):
        return self.model.accuracy(w, self.dataset.X_train, self.dataset.y_train)

    def test_accuracy(self, w):
        return self.model.accuracy(w, self.dataset.X_test, self.dataset.y_test)

    def val_loss(self, w):
        """Loss na validacionom delu (zahteva load_dataset(..., val_size=...))."""
        return self.model.loss(w, self.dataset.X_val, self.dataset.y_val)

    def val_accuracy(self, w):
        return self.model.accuracy(w, self.dataset.X_val, self.dataset.y_val)
