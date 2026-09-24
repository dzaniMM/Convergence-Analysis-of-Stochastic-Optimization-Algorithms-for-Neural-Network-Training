"""Poređenje konvergencije po veličini batch-a i po learning rate-u, za sve optimizatore (moons):
python compare_hyperparams.py [batch|lr] [broj_epoha]
Bez argumenta se crtaju oba panela. Isti MLP/dataset (seed 0, fiksna inicijalizacija) za sve konfiguracije unutar
jednog panela - razlika u krivama dolazi ISKLJUČIVO iz batch_size-a (batch sweep) ili lr-a (lr sweep), ne iz šuma
seed-a. Bazni lr po optimizatoru je isti kao u compare_datasets.make_optimizers() (podešen za batch 32).

Batch sweep: BATCH_SIZES pri baznom lr-u - manji batch je šumovitiji gradijent po koraku, ali više koraka po epohi.
Lr sweep: LR_FACTORS * bazni lr pri batch_size=32.
diverge_factor je veliki da se ceo oblik
krive vidi i kad neka kombinacija divergira, umesto da trening bude rano presečen."""
import copy
import colorsys
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from classification import ClassificationProblem, load_dataset, train_minibatch
from compare_datasets import TRAINING, make_optimizers

BATCH_SIZES = (32, 64, 128, 256, 512)
LR_FACTORS = (1 / 3, 1, 3)
DEFAULT_EPOCHS = TRAINING["moons"]["epochs"]

# fiksna boja po optimizatoru (kategorijalna paleta), dosledna kroz oba panela - identitet optimizatora se NIKAD
# ne menja sa filterom/sweep-om, samo nijansa unutar panela kodira batch_size/lr (videti _shades)
COLORS = {
    "SGD": "#2a78d6", "Momentum": "#eb6834", "NAG": "#1baf7a", "AdaGrad": "#eda100",
    "RMSProp": "#e87ba4", "Adam": "#008300", "AdamW": "#4a3aa7",
}
GRID_COLOR, AXIS_COLOR, MUTED = "#e1e0d9", "#c3c2b7", "#898781"


def _shades(hex_color, n):
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    lightnesses = np.linspace(min(l + 0.30, 0.85), max(l - 0.20, 0.15), n)
    return [colorsys.hls_to_rgb(h, ll, s) for ll in lightnesses]


def run_batch_sweep(name="moons", epochs=DEFAULT_EPOCHS, seed=0):

    problem = ClassificationProblem(load_dataset(name, seed=seed), seed=seed)
    results = {}
    for label, opt in make_optimizers().items():
        results[label] = {bs: train_minibatch(problem, opt, epochs, bs, seed, diverge_factor=1e6)[1]
                          for bs in BATCH_SIZES}
    return problem, results


def run_lr_sweep(name="moons", epochs=DEFAULT_EPOCHS, batch_size=32, seed=0):
    problem = ClassificationProblem(load_dataset(name, seed=seed), seed=seed)
    base_lr, results = {}, {}
    for label, opt in make_optimizers().items():
        base_lr[label] = opt.lr
        results[label] = {}
        for factor in LR_FACTORS:
            opt_f = copy.deepcopy(opt)
            opt_f.lr = opt.lr * factor
            results[label][factor] = train_minibatch(problem, opt_f, epochs, batch_size, seed, diverge_factor=1e6)[1]
    return problem, base_lr, results


def _style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED)


def _plot_sweep(results, sweep_values, legend_fn, title, xlabel, out, would_diverge):
    labels = list(results)
    cols = 4
    rows = -(-len(labels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, label in zip(axes, labels):
        shades = _shades(COLORS[label], len(sweep_values))
        n_diverge = 0
        for color, sweep_val in zip(shades, sweep_values):
            losses = np.clip(results[label][sweep_val], 1e-4, 1e2)
            ax.semilogy(losses, color=color, linewidth=2, label=legend_fn(label, sweep_val), solid_capstyle="round")
            if would_diverge.get((label, sweep_val)):
                n_diverge += 1
        _style_axes(ax)
        suffix = f" ({n_diverge}/{len(sweep_values)} bi divergiralo)" if n_diverge else ""
        ax.set_title(label + suffix, color="#0b0b0b", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[len(labels):]:
        ax.axis("off")
    for ax in axes[-cols:][:len(labels) - (rows - 1) * cols]:
        ax.set_xlabel(xlabel, color=MUTED)
    fig.suptitle(title, color="#0b0b0b", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)


def main_batch_sweep(name="moons", epochs=DEFAULT_EPOCHS):
    print(f"{name}: konvergencija po batch_size-u {BATCH_SIZES} (bazni lr iz make_optimizers(), {epochs} epoha)")
    problem, results = run_batch_sweep(name, epochs)
    would_diverge = {}
    for label, by_bs in results.items():
        for bs, losses in by_bs.items():
            would_diverge[(label, bs)] = bool(np.nanmax(losses) > 10 * losses[0])
        final = {bs: losses[-1] for bs, losses in by_bs.items()}
        print(f"  {label:9s} " + "  ".join(f"batch={bs}: loss={final[bs]:.4f}" for bs in BATCH_SIZES))
    out = f"classification_{name}_batch.png"
    _plot_sweep(results, BATCH_SIZES, lambda label, bs: f"batch={bs}",
                f"{name}: konvergencija po veličini batch-a (bazni lr, seed 0)", "epoha", out, would_diverge)
    print("sačuvano:", out)


def main_lr_sweep(name="moons", epochs=DEFAULT_EPOCHS):
    print(f"{name}: konvergencija po lr-u, faktori {LR_FACTORS} baznog lr-a (batch_size=32, {epochs} epoha)")
    problem, base_lr, results = run_lr_sweep(name, epochs)
    would_diverge = {}
    for label, by_factor in results.items():
        for factor, losses in by_factor.items():
            would_diverge[(label, factor)] = bool(np.nanmax(losses) > 10 * losses[0])
        final = {factor: losses[-1] for factor, losses in by_factor.items()}
        print(f"  {label:9s} (bazni lr={base_lr[label]:g}) " +
              "  ".join(f"{factor:.2g}x: loss={final[factor]:.4f}" for factor in LR_FACTORS))
    out = f"classification_{name}_lr.png"
    _plot_sweep(results, LR_FACTORS, lambda label, factor: f"{factor:.2g}× (lr={factor * base_lr[label]:.3g})",
                f"{name}: konvergencija po learning rate-u (batch_size=32, seed 0)", "epoha", out, would_diverge)
    print("sačuvano:", out)


def main(which=None, n_iters=None):
    epochs = n_iters or DEFAULT_EPOCHS
    if which in (None, "batch"):
        main_batch_sweep(epochs=epochs)
    if which in (None, "lr"):
        main_lr_sweep(epochs=epochs)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("batch", "lr") else None
    rest = sys.argv[2:] if which else sys.argv[1:]
    main(which, int(rest[0]) if rest else None)
