from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    """Klasifikacioni skup podataka podeljen na trening i test deo. Za n_classes=2 je y float 0/1, inače int oznake 0..n_classes-1."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int = 2
    X_val: np.ndarray = None
    y_val: np.ndarray = None


N_CLASSES = 4  # broj klasa za moons/circles
# nivoi težine moons/circles: (skala x, skala y, ugao rotacije u stepenima). Razlikuju se SAMO po linearnoj
# transformaciji obeležja - sve ostalo (klase, šum, broj primera) je isto. easy/hard: skaliranje x-ose s i
# y-ose 1/s, pa je odnos gradijenata po težinama x- i y-ose ~s^2 (1, 100); hard_rotated: isto izduženje kao hard,
# pa rotacija za 45° - loša uslovljenost ostaje, ali više nije poravnata sa osama (x i y postaju jako korelisani,
# istog raspona), pa je ne popravljaju ni standardizacija ni skaliranje koraka po koordinati
LEVELS = {"easy": (1.0, 1.0, 0.0), "hard": (10.0, 0.1, 0.0), "hard_rotated": (10.0, 0.1, 45.0)}


def transform(X, level):
    """Primenjuje LEVELS[level] na X: prvo skaliranje osa, pa rotacija."""
    sx, sy, angle = LEVELS[level]
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (X * np.array([sx, sy])) @ rotation.T


def _moons(n_samples, seed, n_classes=N_CLASSES, noise=0.1):
    """Lanac od n_classes isprepletanih polumeseca (klasa c: luk oko (c, 0.5*(c%2)), naizmenično gore/dole);
    za n_classes=2 i bez skaliranja je to isti oblik kao sklearn make_moons."""
    rng = np.random.default_rng(seed)
    y = np.arange(n_samples) % n_classes
    t = rng.uniform(0, np.pi, n_samples)
    sgn = np.where(y % 2 == 0, 1.0, -1.0)
    X = np.c_[y + sgn * np.cos(t), 0.5 * (y % 2) + sgn * np.sin(t)]
    return X + rng.normal(0, noise, X.shape), y


def _circles(n_samples, seed, n_classes=N_CLASSES, noise=0.05):
    """n_classes koncentričnih prstenova poluprečnika (c+1)/n_classes, klasa c = prsten c."""
    rng = np.random.default_rng(seed)
    y = np.arange(n_samples) % n_classes
    t = rng.uniform(0, 2 * np.pi, n_samples)
    r = (y + 1) / n_classes
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    return X + rng.normal(0, noise, X.shape), y


DATASETS = {"moons": _moons, "circles": _circles}


def load_dataset(name, level="hard", n_samples=2000, test_size=0.25, val_size=0.0, seed=0):
    """2D sintetički skup sa N_CLASSES klasa, obeležja transformisana prema LEVELS[level], stratifikovana podela.
    val_size je udeo CELOG skupa (ne samo trening dela) izdvojen za validaciju; sa val_size=0 nema validacionog dela
    (Dataset.X_val/y_val ostaju None)."""
    X, y = DATASETS[name](n_samples, seed)
    X = transform(X, level)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if not val_size:
        return Dataset(name, X_tr, X_te, y_tr, y_te, n_classes=N_CLASSES)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size / (1 - test_size), random_state=seed, stratify=y_tr)
    return Dataset(name, X_tr, X_te, y_tr, y_te, n_classes=N_CLASSES, X_val=X_val, y_val=y_val)


def load_mnist(n_train=10000, n_test=10000, seed=0):
    """MNIST (28x28 → 784 ulaza skalirano na [0,1], 10 klasa). Zvanična podela 60000/10000, iz svake se uzima slučajan podskup
    (podrazumevano 10000 trening - radi brzine, ~6x kraći trening - i ceo test skup od 10000, koji je jeftin za
    evaluaciju, a drži grešku merenja tačnosti malom). Prvi put se preuzima sa OpenML-a (~15 MB) i kešira u
    ~/scikit_learn_data."""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    rng = np.random.default_rng(seed)
    tr = rng.choice(60000, n_train, replace=False)
    te = 60000 + rng.choice(10000, n_test, replace=False)
    return Dataset("mnist", X[tr] / 255.0, X[te] / 255.0, y[tr].astype(int), y[te].astype(int), n_classes=10)


COVTYPE_CONTINUOUS = 10  # Covertype: prvih 10 obeležja su kontinualna merenja, ostalih 44 su one-hot (0/1)


def load_covtype(standardize=True, n_train=15000, n_test=5000, seed=0):
    """Covertype (fetch_covtype, sklearn): 54 obeležja - 10 kontinualnih (nadmorska visina, nagib, udaljenosti u metrima,
    osenčenost; std do ~1500) i 44 one-hot (tip divljine/zemljišta, 0/1) - i 7 tipova šumskog pokrivača kao klase.
    Slučajan podskup od 581012 primera, stratifikovana podela. standardize=True je SELEKTIVNA standardizacija: samo
    kontinualna obeležja (srednja vrednost/std sa trening dela), one-hot ostaju 0/1; standardize=False su sirova
    obeležja (prirodno, jako loše skalirana)."""
    X, y = fetch_covtype(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_train + n_test, replace=False)
    Xs, ys = X[idx].astype(float), y[idx].astype(int) - 1  # y je 1..7 -> 0..6
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=n_test, random_state=seed, stratify=ys)
    if standardize:
        c = COVTYPE_CONTINUOUS
        mu, sd = X_tr[:, :c].mean(axis=0), X_tr[:, :c].std(axis=0)
        X_tr[:, :c], X_te[:, :c] = (X_tr[:, :c] - mu) / sd, (X_te[:, :c] - mu) / sd
    return Dataset("covtype", X_tr, X_te, y_tr, y_te, n_classes=7)
