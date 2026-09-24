"""Poređenje optimizatora na klasifikaciji (MLP): python compare_datasets.py [moons|circles|mnist|covtype] [nivo/varijanta] [broj_iteracija]
Svi skupovi se treniraju mini-batch-em, broj_iteracija = epohe. Bez argumenta se pokreću samo 2D skupovi
(batch 32, 100 epoha), svaki na sva četiri nivoa težine (LEVELS - skaliranje/rotacija obeležja), uz zbirni grafik
classification_levels.png; nivo može da se zada i eksplicitno. mnist i covtype se traže eksplicitno (veći skupovi, batch
128, do 30 epoha); covtype ima dve varijante (COVTYPE_VARIANTS: selektivno standardizovan i sirov).

Svaki optimizator se pokreće preko više seed-ova (menja podatke/split, init težina MLP-a i redosled mini-batch-eva);
rezultati (tačnost, epoha do cilja) se prijavljuju kao srednja vrednost ± std, ne za jedan seed."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classification import DATASETS, LEVELS, ClassificationProblem, load_covtype, load_dataset, load_mnist, epochs_to_loss, train_minibatch
from optimizers import SGD, NAG, AdaGrad, Adam, AdamW, Momentum, RMSProp

TRAINING = {
    "moons": dict(batch_size=32, epochs=100, tol=1e-3, patience=30, target=0.15, n_seeds=5),
    "circles": dict(batch_size=32, epochs=100, tol=1e-3, patience=30, target=0.1, n_seeds=5),
    "mnist": dict(batch_size=128, epochs=30, tol=1e-3, patience=3, target=0.05, n_seeds=3),
    "covtype": dict(batch_size=128, epochs=30, tol=1e-3, patience=10, target=0.45, n_seeds=3),
}


LEVEL_LABELS = {"easy": "lako", "hard": "teško", "hard_rotated": "teško + rotacija"}

# lr-ovi za moons/circles po nivou težine (mini-batch 32), podešeni mrežom vrednosti (korak 3x, prosek 3 seed-a,
# 100 epoha, najmanji trening loss; svi su unutrašnji optimum). Na easy/hard adaptivni imaju isti lr svuda
# (skaliranje obeležja ih ne dira), a SGD/Momentum/NAG moraju da smanjuju lr kako skala raste; na hard_rotated
# skaliranje po koordinati više ne poništava transformaciju, pa i adaptivni traže drugačiji lr.
LR_2D = {
    "moons": {
        "easy": dict(SGD=0.3, Momentum=0.1, NAG=0.1, AdaGrad=0.3, RMSProp=0.01, Adam=0.01, AdamW=0.01),
        "hard": dict(SGD=0.01, Momentum=0.003, NAG=0.003, AdaGrad=0.3, RMSProp=0.01, Adam=0.01, AdamW=0.01),
        "hard_rotated": dict(SGD=0.01, Momentum=0.003, NAG=0.003, AdaGrad=0.1, RMSProp=0.003, Adam=0.01, AdamW=0.003),
    },
    "circles": {
        "easy": dict(SGD=0.3, Momentum=0.03, NAG=0.1, AdaGrad=0.3, RMSProp=0.01, Adam=0.01, AdamW=0.01),
        "hard": dict(SGD=0.1, Momentum=0.03, NAG=0.03, AdaGrad=0.3, RMSProp=0.01, Adam=0.01, AdamW=0.01),
        "hard_rotated": dict(SGD=0.3, Momentum=0.03, NAG=0.03, AdaGrad=0.3, RMSProp=0.03, Adam=0.01, AdamW=0.01),
    },
}


def make_optimizers(name="moons", level="hard"):
    """Optimizatori sa lr-ovima iz LR_2D[name][level] za moons/circles (mini-batch 32)."""
    lr = LR_2D[name][level]
    return {
        "SGD": SGD(lr=lr["SGD"]),
        "Momentum": Momentum(lr=lr["Momentum"], momentum=0.9),
        "NAG": NAG(lr=lr["NAG"], momentum=0.9),
        "AdaGrad": AdaGrad(lr=lr["AdaGrad"]),
        "RMSProp": RMSProp(lr=lr["RMSProp"]),
        "Adam": Adam(lr=lr["Adam"]),
        "AdamW": AdamW(lr=lr["AdamW"], weight_decay=0.01),
    }


def make_optimizers_mnist():
    """Lr-ovi za MLP 784-128-10 na MNIST podskupu (load_mnist: 10000 trening / 10000 test, mini-batch 128), podešeni
    mrežom vrednosti na punom budžetu (30 epoha, prosek 3 seed-a, najmanji trening loss; svi unutrašnji optimum).
    Skriveni sloj od 128 neurona je biran poređenjem širina/dubina na celom skupu (60000): 128 daje +0.6pp testa
    naspram 64, a šire/dublje mreže još samo do ~0.2pp uz 2-5x duži trening."""
    return {
        "SGD": SGD(lr=1.0),
        "Momentum": Momentum(lr=0.3, momentum=0.9),
        "NAG": NAG(lr=0.3, momentum=0.9),
        "AdaGrad": AdaGrad(lr=0.1),
        "RMSProp": RMSProp(lr=0.01),
        "Adam": Adam(lr=0.01),
        "AdamW": AdamW(lr=0.01, weight_decay=0.01),
    }


# Covertype: dve varijante istog podskupa (15000/5000) - selektivno standardizovan (samo 10 kontinualnih obeležja, one-hot
# ostaju 0/1) i sirov (kontinualna obeležja u originalnim jedinicama, std do ~1500 - prirodno jako loše skalirano).
# Mreža 54-256-128-7 sa ReLU (He init), bez BatchNorm-a (on bi sam normalizovao ulaze slojeva i sakrio efekat skala).
COVTYPE_VARIANTS = {"standardized": "selektivno standardizovano", "raw": "sirovo"}
COVTYPE_LAYERS = (54, 256, 128, 7)
# lr-ovi podešeni mrežom vrednosti (korak ~3x, 30 epoha, prosek 3 seed-a, najmanji trening loss; svi unutrašnji)
LR_COVTYPE = {
    "standardized": dict(SGD=1.0, Momentum=0.1, NAG=0.1, AdaGrad=0.1, RMSProp=0.003, Adam=0.003, AdamW=0.003),
    "raw": dict(SGD=1e-4, Momentum=1e-5, NAG=1e-5, AdaGrad=0.1, RMSProp=0.001, Adam=0.003, AdamW=0.003),
}


def make_optimizers_covtype(variant="standardized"):
    """Optimizatori sa lr-ovima iz LR_COVTYPE[variant] (mini-batch 128)."""
    lr = LR_COVTYPE[variant]
    return {
        "SGD": SGD(lr=lr["SGD"]),
        "Momentum": Momentum(lr=lr["Momentum"], momentum=0.9),
        "NAG": NAG(lr=lr["NAG"], momentum=0.9),
        "AdaGrad": AdaGrad(lr=lr["AdaGrad"]),
        "RMSProp": RMSProp(lr=lr["RMSProp"]),
        "Adam": Adam(lr=lr["Adam"]),
        "AdamW": AdamW(lr=lr["AdamW"], weight_decay=0.01),
    }


def run(name, n_iters=None, seed=0, level="hard"):
    """Trenira isti MLP (ista početna tačka) svakim optimizatorom. Vraća problem i rezultate po optimizatoru.
    level je nivo težine moons/circles, odnosno varijanta covtype-a (COVTYPE_VARIANTS); za mnist se ignoriše.
    Mini-batch trening, n_iters je najveći broj epoha (podrazumevano iz TRAINING); staje ranije na platou loss-a
    ili pri divergenciji. `to_target` je epoha u kojoj je loss prvi put pao na cilj (None ako nikad)."""
    cfg = TRAINING[name]
    if name == "mnist":
        problem = ClassificationProblem(load_mnist(seed=seed), layers=(784, 128, 10), seed=seed)
        optimizers = make_optimizers_mnist()
    elif name == "covtype":
        problem = ClassificationProblem(load_covtype(standardize=(level == "standardized"), seed=seed),
                                        layers=COVTYPE_LAYERS, seed=seed, activation="relu")
        optimizers = make_optimizers_covtype(level)
    else:
        problem = ClassificationProblem(load_dataset(name, level=level, seed=seed), seed=seed)
        optimizers = make_optimizers(name, level)
    results = {}
    for label, opt in optimizers.items():
        w, vals, stop = train_minibatch(problem, opt, n_iters or cfg["epochs"], cfg["batch_size"], seed,
                                        cfg["tol"], cfg["patience"])
        results[label] = dict(w=w, losses=vals, stop=stop, to_target=epochs_to_loss(vals, cfg["target"]),
                              train_acc=problem.train_accuracy(w), test_acc=problem.test_accuracy(w))
    return problem, results


def run_seeds(run_fn, *args, n_seeds=5, **kwargs):
    """Poziva run_fn(*args, seed=s, **kwargs) za s=0..n_seeds-1 (run_fn je npr. run()).
    Vraća (problems, results_list) - liste po seed-u, istim redosledom."""
    runs = [run_fn(*args, seed=s, **kwargs) for s in range(n_seeds)]
    problems, results_list = zip(*runs)
    return list(problems), list(results_list)


def aggregate(results_list):
    """Preko liste rezultata (jedan po seed-u, isti optimizatori u svakom) računa srednju vrednost i std. devijaciju
    test/trening tačnosti po optimizatoru, i srednju epohu-do-cilja RAČUNATU SAMO nad seed-ovima koji su ga
    dostigli (uz broj takvih seed-ova - `to_target_hits`)."""
    agg = {}
    for label in results_list[0]:
        test_acc = np.array([r[label]["test_acc"] for r in results_list])
        train_acc = np.array([r[label]["train_acc"] for r in results_list])
        hits = [r[label]["to_target"] for r in results_list if r[label]["to_target"] is not None]
        agg[label] = dict(
            test_acc_mean=test_acc.mean(), test_acc_std=test_acc.std(),
            train_acc_mean=train_acc.mean(), train_acc_std=train_acc.std(),
            to_target_mean=(float(np.mean(hits)) if hits else None), to_target_hits=len(hits),
            n_seeds=len(results_list),
        )
    return agg


def plot_accuracy(name, results, agg, out, batch_size):
    """Za skupove sa više od 2 ulaza nema granice odluke: loss kroz epohe (seed 0, radi oblika) i trening/test
    tačnost po optimizatoru (srednja vrednost ± std preko seed-ova, error bar-ovi)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, r in results.items():
        ax1.semilogy(r["losses"], label=label)
    ax1.set(title=f"{name}: trening loss, seed 0 (mini-batch {batch_size})", xlabel="epoha")
    ax1.legend()
    x = np.arange(len(results))
    n = agg[next(iter(agg))]["n_seeds"]
    train_means = [agg[l]["train_acc_mean"] for l in results]
    test_means = [agg[l]["test_acc_mean"] for l in results]
    ax2.bar(x - 0.2, train_means, 0.4, yerr=[agg[l]["train_acc_std"] for l in results], label="trening")
    ax2.bar(x + 0.2, test_means, 0.4, yerr=[agg[l]["test_acc_std"] for l in results], label="test")
    # dinamičan donji limit (umesto fiksnog 0.9), prati najnižu tačnost
    ylo = max(0.0, min(train_means + test_means) - 0.1)
    ax2.set(title=f"{name}: tačnost (srednje ± std, {n} seed-ova)", xticks=x, xticklabels=list(results), ylim=(ylo, 1.0))
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def plot(name, problem, results, agg, out, title=None):
    """Granica odluke je za seed 0 (jedan reprezentativan trening); naslov svakog panela pokazuje srednju test
    tačnost ± std preko seed-ova (ne samo za taj jedan prikazani trening)."""
    ds = problem.dataset
    X = np.vstack([ds.X_train, ds.X_test])
    pad = 0.05 * (X.max(axis=0) - X.min(axis=0))  # po osi - obeležja su namerno na različitim skalama
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], 200),
                         np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], 200))
    grid = np.c_[gx.ravel(), gy.ravel()]

    n = len(results) + 1
    cols = 4
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(4 * cols, 4 * -(-n // cols)))
    axes = axes.ravel()
    for label, r in results.items():
        axes[0].semilogy(r["losses"], label=label)
    axes[0].set(title=f"{title or name}: trening loss, seed 0 (mini-batch {TRAINING[name]['batch_size']})", xlabel="epoha")
    axes[0].legend()

    for ax, (label, r) in zip(axes[1:], results.items()):
        zz = problem.model.predict(r["w"], grid).reshape(gx.shape)
        k = ds.n_classes
        ax.contourf(gx, gy, zz, levels=np.arange(k + 1) - 0.5, alpha=0.25, cmap="tab10", vmin=0, vmax=9)
        ax.scatter(*ds.X_train.T, c=ds.y_train, cmap="tab10", vmin=0, vmax=9, s=6, edgecolors="none")
        ax.scatter(*ds.X_test.T, c=ds.y_test, cmap="tab10", vmin=0, vmax=9, s=14, edgecolors="k", linewidths=0.4)
        ax.set(title=f"{label} (test acc {agg[label]['test_acc_mean']:.1%} ± {agg[label]['test_acc_std']:.1%})")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def plot_dataset_levels(out, names=("moons", "circles"), seed=0):
    """Isti skupovi (isti seed, iste tačke) na svim nivoima težine, u PRAVIM razmerama (ista jedinica na obe ose):
    što je nivo teži, to je skup spljošteniji - teški nivo je vodoravna traka, a rotirani ista traka okrenuta
    dijagonalno. Uz nivoe sa skalom >= 10 je i uvećan pogled popreko trake (tačke vraćene u neorotirane koordinate,
    ose nezavisno razvučene) - oblik klasa je i dalje tu, samo spljošten."""
    fig, axes = plt.subplots(len(names), len(LEVELS), figsize=(4.5 * len(LEVELS), 4.2 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        for col, (level, (sx, sy, angle)) in enumerate(LEVELS.items()):
            ds = load_dataset(name, level=level, seed=seed)
            ax = axes[row, col]
            ax.scatter(*ds.X_train.T, c=ds.y_train, cmap="tab10", vmin=0, vmax=9, s=4, edgecolors="none")
            ax.set_aspect("equal", adjustable="datalim")
            rot = f", rot {angle:g}°" if angle else ""
            ax.set_title(f"{name}, {LEVEL_LABELS[level]} (x×{sx:g}, y×{sy:.3g}{rot})", fontsize=10)
            if sx / sy >= 100:
                a = np.deg2rad(angle)
                rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                along, across = (ds.X_train @ rotation).T  # inverzna rotacija: koordinate duž i popreko trake
                inset = ax.inset_axes([0.04, 0.56, 0.42, 0.4])
                inset.scatter(along, across, c=ds.y_train, cmap="tab10", vmin=0, vmax=9, s=1, edgecolors="none")
                inset.set_xticks([])
                inset.set_yticks([])
                inset.set_title("popreko trake, uvećano", fontsize=7, pad=2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def plot_levels(aggs, out):
    """Zbirni grafik nivoa težine: po skupu (kolone) test tačnost i epoha do cilja u zavisnosti od nivoa, jedna
    linija po optimizatoru (boje u istom redosledu kao krive loss-a u plot()). aggs = {(name, level): aggregate(...)}."""
    names = list(dict.fromkeys(n for n, _ in aggs))
    levels = [lv for lv in LEVELS if any((n, lv) in aggs for n in names)]
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 8), squeeze=False)
    x = np.arange(len(levels))
    for col, name in enumerate(names):
        ax_acc, ax_ep = axes[0, col], axes[1, col]
        for label in aggs[(name, levels[0])]:
            acc = np.array([aggs[(name, lv)][label]["test_acc_mean"] for lv in levels])
            std = np.array([aggs[(name, lv)][label]["test_acc_std"] for lv in levels])
            ax_acc.errorbar(x, acc, yerr=std, marker="o", capsize=3, label=label)
            ep = [aggs[(name, lv)][label]["to_target_mean"] for lv in levels]
            ax_ep.plot(x, [np.nan if e is None else e for e in ep], marker="o", label=label)
        target = TRAINING[name]["target"]
        ax_acc.set(title=f"{name}: test tačnost po nivou težine", ylabel="test tačnost",
                   xticks=x, xticklabels=[LEVEL_LABELS[lv] for lv in levels])
        ax_ep.set(title=f"{name}: epoha do loss ≤ {target} (prazno = nijedan seed nije stigao)",
                  ylabel="epoha", xticks=x, xticklabels=[LEVEL_LABELS[lv] for lv in levels])
        ax_acc.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main(names=None, n_iters=None, levels=None):
    """levels: nivoi za moons/circles (LEVELS), odnosno varijante za covtype (COVTYPE_VARIANTS); podrazumevano sve."""
    aggs = {}
    for name in names or DATASETS:
        cfg = TRAINING[name]
        if name == "mnist":
            name_levels = [None]
        elif name == "covtype":
            name_levels = [lv for lv in levels or COVTYPE_VARIANTS if lv in COVTYPE_VARIANTS]
        else:
            name_levels = [lv for lv in levels or LEVELS if lv in LEVELS]
        for level in name_levels:
            kwargs = {} if level is None else {"level": level}
            problems, results_list = run_seeds(run, name, n_iters, n_seeds=cfg["n_seeds"], **kwargs)
            problem, results = problems[0], results_list[0]
            agg = aggregate(results_list)
            label_of = COVTYPE_VARIANTS if name == "covtype" else LEVEL_LABELS
            title = name if level is None else f"{name} ({label_of[level]})"
            print(f"{title}: {len(problem.dataset.X_train)} trening / {len(problem.dataset.X_test)} test, "
                  f"{problem.model.n_params} parametara, najviše {n_iters or cfg['epochs']} epoha (batch {cfg['batch_size']}), "
                  f"cilj loss <= {cfg['target']}, {cfg['n_seeds']} seed-ova")
            for label, r in results.items():
                a = agg[label]
                hit = "-" if a["to_target_mean"] is None else f"{a['to_target_mean']:.0f} ({a['to_target_hits']}/{a['n_seeds']} seed-ova)"
                print(f"  {label:9s} train={a['train_acc_mean']:.3f}±{a['train_acc_std']:.3f}  "
                      f"test={a['test_acc_mean']:.3f}±{a['test_acc_std']:.3f}  "
                      f"seed0: loss={r['losses'].min():.4f} stao={len(r['losses']) - 1:3d} ({r['stop']})  "
                      f"epoha do cilja={hit}")
            if problem.dataset.X_train.shape[1] != 2:
                out = f"classification_{name}.png" if level is None else f"classification_{name}_{level}.png"
                plot_accuracy(title, results, agg, out, batch_size=cfg["batch_size"])
            else:
                aggs[(name, level)] = agg
                out = f"classification_{name}_{level}.png"
                plot(name, problem, results, agg, out, title)
            print("sačuvano:", out)
    if len({lv for _, lv in aggs}) > 1:
        plot_levels(aggs, "classification_levels.png")
        plot_dataset_levels("classification_datasets.png", names=list(dict.fromkeys(n for n, _ in aggs)))
        print("sačuvano: classification_levels.png, classification_datasets.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    names = [a for a in args if a in DATASETS or a in ("mnist", "covtype")] or None
    levels = [a for a in args if a in LEVELS or a in COVTYPE_VARIANTS] or None
    n_iters = next((int(a) for a in args if a.isdigit()), None)
    main(names, n_iters, levels)
