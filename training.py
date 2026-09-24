"""Generička mini-batch trening petlja - ne zna ništa o klasifikaciji, radi nad bilo kojim
problemom koji ima `.w0`, `.n_train`, `.f(w)` i `.grad(w, idx=None)` (npr. ClassificationProblem)."""
import time

import numpy as np


def train_minibatch(problem, opt, epochs=100, batch_size=32, seed=0, tol=None, patience=5, diverge_factor=10.0,
                     w0=None, warm_start=False, times=None):
    """Mini-batch trening: svake epohe se trening skup promeša i podeli na batch-eve, a svaki batch daje jedan korak.
    NAG dobija grad_fn nad istim batch-om.

    Kriterijumi zaustavljanja (samo nad trening loss-om celog skupa, računatim posle svake epohe):
      plato:       ako je tol zadat, staje kad loss nijednu od poslednjih `patience` epoha nije pao ispod best*(1 - tol)
      divergencija: staje kad loss nije konačan ili je veći od diverge_factor * početni loss
      epochs:      najveći broj epoha
    Vraća (w, losses, stop): w su težine iz epohe sa najmanjim loss-om, losses je loss pre treninga i posle svake
    odrađene epohe (dužine odrađenih epoha + 1), a stop je "epochs", "plateau" ili "divergence".

    w0/warm_start: za nastavak već započetog treninga - w0 je početna tačka (podrazumevano
    problem.w0), a warm_start=True preskače opt.reset() i tako čuva stanje optimizatora (isti objekat opt) iz
    prethodnog poziva.

    times: opciono, lista u koju se upisuje kumulativno vreme (s) SAMO koraka optimizacije (gradijenti + opt.step,
    bez računanja loss-a celog skupa posle epohe, koje je isto za sve optimizatore) - 0.0 pa po jedna vrednost
    posle svake odrađene epohe, poravnato sa losses."""
    if not warm_start:
        opt.reset()
    rng = np.random.default_rng(seed)
    n = problem.n_train
    w = (problem.w0 if w0 is None else w0).copy()
    losses = [problem.f(w)]
    best_w, best, stale, stop = w.copy(), losses[0], 0, "epochs"
    elapsed = 0.0
    if times is not None:
        times.append(elapsed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            grad_fn = lambda x, idx=idx: problem.grad(x, idx)
            w = opt.step(w, grad_fn(w), grad_fn)
        elapsed += time.perf_counter() - t0
        if times is not None:
            times.append(elapsed)
        losses.append(problem.f(w))
        if not np.isfinite(losses[-1]) or losses[-1] > diverge_factor * losses[0]:
            stop = "divergence"
            break
        if losses[-1] < best * (1 - (tol or 0)):
            stale = 0
        else:
            stale += 1
        if losses[-1] < best:
            best, best_w = losses[-1], w.copy()
        if tol is not None and stale >= patience:
            stop = "plateau"
            break
    return best_w, np.array(losses), stop


def epochs_to_loss(losses, target):
    """Prva epoha u kojoj je loss <= target, ili None ako ga optimizator nikad nije dostigao."""
    hit = np.flatnonzero(np.asarray(losses) <= target)
    return int(hit[0]) if len(hit) else None
