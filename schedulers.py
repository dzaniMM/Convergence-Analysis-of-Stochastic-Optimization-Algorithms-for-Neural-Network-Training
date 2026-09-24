"""Rasporedi opadanja learning rate-a: svaka fabrika vraća funkciju `epoch -> lr`, za `train_minibatch(...,
schedule=...)` (koji pre svake epohe postavlja `opt.lr = schedule(epoch)`). Generički - ne zna ništa o
optimizatorima, klasifikaciji ni regresiji.

Zašto je ovo bitno: konstantan lr uz mini-batch šum ne konvergira tačno u minimum, nego u "šum-lopticu" oko
njega (videti regression.png) - klasična stohastička aproksimacija (Robbins-Monro) zahteva da korak opada
(sum(lr_t) = beskonačno, sum(lr_t^2) < beskonačno) da bi se konvergiralo tačno. AdaGrad ovo dvec ima
(G raste, pa lr/sqrt(G) sam opada); SGD/Momentum/RMSProp/Adam sa fiksnim lr - ne."""
import numpy as np


def constant(lr0):
    """Bez opadanja - isto ponašanje kao da schedule uopšte nije prosleđen."""
    return lambda epoch: lr0


def step_decay(lr0, drop=0.5, every=50):
    """lr0 * drop^floor(epoch/every) - naglo opadanje na svakih `every` epoha (standardno u praksi)."""
    return lambda epoch: lr0 * drop ** (epoch // every)


def inverse_time(lr0, decay=0.01):
    """lr0 / (1 + decay*epoch) - klasičan O(1/t) raspored; zadovoljava Robbins-Monro uslove
    (sum(lr_t) diverguje, sum(lr_t^2) konvergira), pa SGD uz njega teorijski konvergira tačno u minimum."""
    return lambda epoch: lr0 / (1 + decay * epoch)


def cosine(lr0, total_epochs, min_lr=0.0):
    """Kosinusno opadanje od lr0 (epoha 0) do min_lr (epoha total_epochs), posle čega ostaje na min_lr."""
    def schedule(epoch):
        t = min(epoch, total_epochs)
        return min_lr + 0.5 * (lr0 - min_lr) * (1 + np.cos(np.pi * t / total_epochs))
    return schedule
