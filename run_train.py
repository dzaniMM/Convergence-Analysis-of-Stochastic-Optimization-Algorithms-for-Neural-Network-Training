"""Trening jednog optimizatora prema config.yaml: python run_train.py [config.yaml]
Čita dataset/model/optimizer/training iz YAML-a (videti config.yaml za sve ključeve i
podrazumevane vrednosti), pokreće jedan train_minibatch poziv i ispisuje/crta rezultat - "ručni"
ekvivalent jednog panela iz compare_datasets.py, kad treba samo JEDNA kombinacija bez poređenja.

Sekcija optimizer prosleđuje sve ključeve osim "name" direktno konstruktoru (optimizers/*.py), dataset sekcija
isto tako direktno u load_dataset/load_mnist - nema posebne validacije, pogrešan ključ puca kao TypeError iz
odgovarajuće funkcije."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from classification import ClassificationProblem, load_covtype, load_dataset, load_mnist, train_minibatch
from optimizers import SGD, NAG, AdaGrad, Adam, AdamW, Momentum, RMSProp

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "nag": NAG, "adagrad": AdaGrad,
              "rmsprop": RMSProp, "adam": Adam, "adamw": AdamW}
LOADERS = {"mnist": load_mnist, "covtype": load_covtype}  # moons/circles idu preko load_dataset


def build_problem(dataset_cfg, model_cfg, seed):
    dataset_cfg = dict(dataset_cfg)
    name = dataset_cfg.pop("name")
    loader = LOADERS.get(name)
    dataset = loader(seed=seed, **dataset_cfg) if loader else load_dataset(name, seed=seed, **dataset_cfg)
    return ClassificationProblem(dataset, layers=tuple(model_cfg["layers"]), seed=seed,
                                 activation=model_cfg.get("activation", "tanh"))


def build_optimizer(optimizer_cfg):
    optimizer_cfg = dict(optimizer_cfg)
    name = optimizer_cfg.pop("name").lower()
    return OPTIMIZERS[name](**optimizer_cfg)


def plot_loss(losses, cfg, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(losses)
    ax.set(title=f"{cfg['dataset']['name']}: {cfg['optimizer']['name']} (trening loss)",
           xlabel="epoha", ylabel="trening loss")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main(config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed = cfg.get("seed", 0)
    problem = build_problem(cfg["dataset"], cfg["model"], seed)
    opt = build_optimizer(cfg["optimizer"])
    train_cfg = cfg.get("training", {})
    w, losses, stop = train_minibatch(
        problem, opt,
        epochs=train_cfg.get("epochs", 100),
        batch_size=train_cfg.get("batch_size", 32),
        seed=seed,
        tol=train_cfg.get("tol"),
        patience=train_cfg.get("patience", 5),
        diverge_factor=train_cfg.get("diverge_factor", 10.0),
    )
    print(f"{cfg['dataset']['name']} / {cfg['optimizer']['name']}: stao={stop} posle {len(losses) - 1} epoha")
    print(f"  trening loss: {losses[0]:.4f} -> {losses[-1]:.4f} (min {losses.min():.4f})")
    print(f"  trening tačnost={problem.train_accuracy(w):.3f}  test tačnost={problem.test_accuracy(w):.3f}")
    out = (cfg.get("output") or {}).get("plot")
    if out:
        plot_loss(losses, cfg, out)
        print("sačuvano:", out)
    return problem, w, losses, stop


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
