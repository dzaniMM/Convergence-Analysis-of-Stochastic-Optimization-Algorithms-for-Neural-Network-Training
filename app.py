"""Interaktivna aplikacija: zadavanje parametara treninga (dataset/optimizator/hiperparametri) i poređenje
konvergencije više treninga na istom grafiku. Pokretanje:
    venv/bin/streamlit run app.py
Levi meni pokreće JEDAN trening (isti tok kao run_train.py: build_optimizer odatle),
rezultat se dodaje na listu; glavni panel iscrtava trening loss (log skala) za izabrane treninge i
tabelu sa tačnošću. Boja je fiksna po optimizatoru (COLORS iz compare_hyperparams.py, ista paleta
kao ostatak projekta); više treninga istog optimizatora (npr. različit lr) dobija nijanse iste boje
preko _shades, sortirano po lr (isti obrazac sekvencijalnog kodiranja kao compare_hyperparams.py).
Vreme treninga se meri u train_minibatch(times=...) - samo koraci optimizacije, bez računanja loss-a celog
skupa posle epohe; x-osa grafika može biti epoha ili to vreme."""
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classification import ClassificationProblem, load_dataset, train_minibatch
from compare_datasets import COVTYPE_VARIANTS, LEVEL_LABELS, LR_2D, LR_COVTYPE
from compare_hyperparams import AXIS_COLOR, COLORS, GRID_COLOR, MUTED, _shades
from run_train import LOADERS, OPTIMIZERS, build_optimizer

COLOR_KEY = {"sgd": "SGD", "momentum": "Momentum", "nag": "NAG", "adagrad": "AdaGrad",
             "rmsprop": "RMSProp", "adam": "Adam", "adamw": "AdamW"}
# lr-ovi podešeni mrežom vrednosti za svaki dataset/nivo (moons/circles: compare_datasets.LR_2D, covtype:
# compare_datasets.LR_COVTYPE, mnist: isto kao make_optimizers_mnist()) - NE sirovi podrazumevani lr-ovi iz konstruktora optimizatora (Adam(lr=0.001) i sl.),
# koji su i do 10x manji od podešenih i daju lažan utisak da je neki optimizator "mnogo bolji" samo zato što
# ostali kreću sa prespornim lr-om.
MNIST_LR = {"sgd": 1.0, "momentum": 0.3, "nag": 0.3, "adagrad": 0.1, "rmsprop": 0.01, "adam": 0.01, "adamw": 0.01}


def _default_lr(dataset_name, level, optimizer_name):
    if dataset_name == "mnist":
        return MNIST_LR[optimizer_name]
    if dataset_name == "covtype":
        return LR_COVTYPE[level][COLOR_KEY[optimizer_name]]
    return LR_2D[dataset_name][level][COLOR_KEY[optimizer_name]]
# skriveni slojevi po dataset-u, FIKSNO - isto kao compare_datasets.py (nije podesivo u aplikaciji,
# jer bi promena arhitekture bez ponovnog podešavanja lr-a gore dala varljivo poređenje, kao ranije
# sa pogrešnim podrazumevanim lr-ovima)
HIDDEN_LAYERS = {"moons": (16, 16), "circles": (16, 16), "mnist": (128,), "covtype": (256, 128)}
ACTIVATION = {"covtype": "relu"}  # ostali skupovi: tanh

st.set_page_config(page_title="Optimizacija - konvergencija", layout="wide")


@st.cache_data(show_spinner="Učitavanje dataset-a...")
def _load_dataset(name, seed, **kwargs):
    loader = LOADERS.get(name)
    return loader(seed=seed, **kwargs) if loader else load_dataset(name, seed=seed, **kwargs)


def _build_layers(dataset, hidden):
    input_dim = dataset.X_train.shape[1]
    output_dim = 1 if dataset.n_classes == 2 else dataset.n_classes
    return (input_dim, *hidden, output_dim)


st.sidebar.header("Parametri treninga")
dataset_name = st.sidebar.selectbox("Dataset", ["moons", "circles", "mnist", "covtype"])
level, level_labels = None, None
if dataset_name == "covtype":
    level_labels = COVTYPE_VARIANTS
    level = st.sidebar.selectbox("Varijanta", list(level_labels), format_func=level_labels.get)
elif dataset_name != "mnist":
    level_labels = LEVEL_LABELS
    level = st.sidebar.selectbox("Nivo težine", list(level_labels), index=2, format_func=level_labels.get)
optimizer_name = st.sidebar.selectbox("Optimizator", list(OPTIMIZERS.keys()))

with st.sidebar.form("run_form"):
    # veličina/podela dataset-a NIJE podesiva ovde - uvek ceo skup, fiksno podeljen isto kao
    # svuda drugde u projektu (load_dataset/load_mnist/load_covtype podrazumevane vrednosti):
    # moons/circles 2000 uzoraka / test_size 0.25 (1500/500), mnist 10000/10000, covtype 15000/5000
    if dataset_name == "covtype":
        dataset_kwargs = {"standardize": level == "standardized"}
    else:
        dataset_kwargs = {} if level is None else {"level": level}

    lr = st.number_input("lr", min_value=0.0, value=_default_lr(dataset_name, level, optimizer_name), format="%.5f")
    optimizer_kwargs = {"lr": lr}
    if optimizer_name in ("momentum", "nag"):
        optimizer_kwargs["momentum"] = st.slider("momentum", 0.0, 0.999, 0.9)
        if optimizer_name == "momentum":
            optimizer_kwargs["nesterov"] = st.checkbox("nesterov", False)
    elif optimizer_name == "adagrad":
        optimizer_kwargs["eps"] = st.number_input("eps", value=1e-8, format="%.1e")
    elif optimizer_name == "rmsprop":
        optimizer_kwargs["rho"] = st.slider("rho", 0.5, 0.999, 0.9)
        optimizer_kwargs["eps"] = st.number_input("eps", value=1e-8, format="%.1e")
    elif optimizer_name in ("adam", "adamw"):
        optimizer_kwargs["beta1"] = st.slider("beta1", 0.5, 0.999, 0.9)
        optimizer_kwargs["beta2"] = st.slider("beta2", 0.9, 0.9999, 0.999)
        optimizer_kwargs["eps"] = st.number_input("eps", value=1e-8, format="%.1e")
        if optimizer_name == "adamw":
            optimizer_kwargs["weight_decay"] = st.number_input("weight_decay", value=0.01, format="%.4f")

    batch_size = st.number_input("batch_size", 1, 2048, 32)
    epochs = st.number_input("epochs", 1, 2000, 200)
    use_tol = st.checkbox("Plato kriterijum (tol)", True)
    tol = st.number_input("tol", value=0.001, min_value=0.0, format="%.5f") if use_tol else None
    patience = st.number_input("patience", 1, 200, 30)
    diverge_factor = st.number_input("diverge_factor", value=10.0, min_value=1.0)
    seed = st.number_input("seed", 0, 10_000, 0)
    label = st.text_input("Naziv treninga (opciono)")

    submitted = st.form_submit_button("Pokreni trening")

if submitted:
    try:
        with st.spinner("Trening u toku..."):
            dataset = _load_dataset(dataset_name, seed, **dataset_kwargs)
            layers = _build_layers(dataset, HIDDEN_LAYERS[dataset_name])
            problem = ClassificationProblem(dataset, layers=layers, seed=seed,
                                            activation=ACTIVATION.get(dataset_name, "tanh"))
            opt = build_optimizer({"name": optimizer_name, **optimizer_kwargs})
            times = []
            w, losses, stop = train_minibatch(
                problem, opt, epochs=epochs, batch_size=batch_size, seed=seed,
                tol=tol, patience=patience, diverge_factor=diverge_factor,
                times=times,
            )
        run = {
            "label": label or f"{dataset_name}{'' if level is None else '-' + level_labels[level]}/{optimizer_name}/lr={lr:g}",
            "losses": losses,
            "times": times,
            "stop": stop,
            "optimizer_name": optimizer_name,
            "lr": lr,
            "train_acc": problem.train_accuracy(w),
            "test_acc": problem.test_accuracy(w),
        }
        st.session_state.setdefault("runs", []).append(run)
    except Exception as exc:
        st.error(f"Trening nije uspeo: {exc}")

st.title("Konvergencija optimizatora")
runs = st.session_state.get("runs", [])

if not runs:
    st.info("Podesi parametre u meniju sa leve strane i klikni 'Pokreni trening'.")
else:
    labels = [r["label"] for r in runs]
    selected = st.multiselect("Treninzi za prikaz", labels, default=labels)
    x_axis = st.segmented_control("X-osa", ["epoha", "vreme (s)"], default="epoha", required=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED)

    by_optimizer = {}
    for r in runs:
        if r["label"] in selected:
            by_optimizer.setdefault(r["optimizer_name"], []).append(r)
    for opt_name, opt_runs in by_optimizer.items():
        opt_runs.sort(key=lambda r: r["lr"])
        shades = _shades(COLORS[COLOR_KEY[opt_name]], len(opt_runs))
        for color, r in zip(shades, opt_runs):
            x = r["times"] if x_axis == "vreme (s)" else range(len(r["losses"]))
            ax.semilogy(x, r["losses"], color=color, linewidth=2, label=r["label"], solid_capstyle="round")

    ax.set(xlabel="epoha" if x_axis == "epoha" else "vreme treninga (s, samo koraci optimizacije)",
           ylabel="trening loss (log skala)")
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe([
        {"trening": r["label"], "optimizator": r["optimizer_name"], "stop": r["stop"],
         "epohe": len(r["losses"]) - 1, "trening tačnost": round(r["train_acc"], 3),
         "test tačnost": round(r["test_acc"], 3),
         "vreme (s)": round(r["times"][-1], 2),
         "ms/epohi": round(1000 * r["times"][-1] / max(1, len(r["losses"]) - 1), 1)}
        for r in runs
    ], width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        to_remove = st.multiselect("Ukloni treninge", labels, key="to_remove")
        if st.button("Ukloni izabrane") and to_remove:
            st.session_state["runs"] = [r for r in runs if r["label"] not in to_remove]
            st.rerun()
    with col2:
        if st.button("Obriši sve treninge"):
            st.session_state["runs"] = []
            st.rerun()
