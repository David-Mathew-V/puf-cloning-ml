"""Makes the plots in figures/ from the CSVs in results/.

    python experiments/make_figures.py
"""
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
MODEL_STYLE = {  # color, marker, label
    "LR":         ("#2a78d6", "o", "Logistic regression"),
    "MLP":        ("#eb6834", "s", "MLP"),
    "PLR":        ("#1baf7a", "D", "Product-LR (model-aware)"),
    "Perceptron": ("#eda100", "^", "Perceptron"),
    "RBFSVM":     ("#e87ba4", "v", "RBF-kernel SVM"),
    "LinSVM":     ("#008300", "P", "Linear SVM"),
}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5.5, "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def agg(df, keys):
    g = df.groupby(keys)["acc"].agg(["mean", "std", "count"]).reset_index()
    t = df.groupby(keys)["train_time_s"].mean().reset_index(name="time")
    return g.merge(t, on=keys)


def crps_to_reach(Ns, accs, target):
    # interpolate (in log N) where the mean accuracy first crosses target
    Ns, accs = np.asarray(Ns, float), np.asarray(accs, float)
    for i, a in enumerate(accs):
        if a >= target:
            if i == 0:
                return float(Ns[0])
            a0, a1, n0, n1 = accs[i - 1], a, np.log(Ns[i - 1]), np.log(Ns[i])
            return float(np.exp(n0 + (target - a0) * (n1 - n0) / (a1 - a0)))
    return None


def plot_model_curves(ax, g, models, raw=None):
    for m in models:
        c, mk, lab = MODEL_STYLE[m]
        d = g[(g.model == m) & (g.features == "phi")].sort_values("N")
        if len(d):
            ax.plot(d.N, d["mean"] * 100, color=c, marker=mk, label=lab + (" on Φ" if raw else ""))
            ax.fill_between(d.N, (d["mean"] - d["std"]) * 100, (d["mean"] + d["std"]) * 100,
                            color=c, alpha=0.12, lw=0)
    for m in raw or []:
        c, mk, lab = MODEL_STYLE[m]
        d = g[(g.model == m) & (g.features == "raw")].sort_values("N")
        if len(d):
            ax.plot(d.N, d["mean"] * 100, color=c, marker=mk, ls="--", mfc="white",
                    label=lab + " on raw bits")


def finish_acc_axis(ax, title=None):
    ax.set_xscale("log")
    ax.axhline(95, color=INK2, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[0], 95.6, " 95 %", color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(45, 101)
    ax.set_xlabel("Training CRPs available to the attacker (log scale)")
    ax.set_ylabel("Prediction accuracy on unseen challenges (%)")
    if title:
        ax.set_title(title, loc="left", color=INK)


summary = {}

def fig_e1():
    df = load("e1_arbiter.csv")
    if df is None:
        return
    g = agg(df, ["model", "features", "N"])
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    plot_model_curves(ax, g, ["Perceptron", "LR", "LinSVM", "RBFSVM", "MLP"], raw=["LR", "RBFSVM"])
    finish_acc_axis(ax, "64-bit Arbiter PUF: every linear model breaks it once features are Φ")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8.2, ncol=1)
    fig.savefig(os.path.join(FIG, "fig_e1_arbiter.png")); plt.close(fig)
    s = {}
    for (m, f), d in g.groupby(["model", "features"]):
        d = d.sort_values("N")
        s[f"{m}_{f}"] = dict(N95=crps_to_reach(d.N, d["mean"], 0.95), N99=crps_to_reach(d.N, d["mean"], 0.99),
                             acc={int(n): round(a * 100, 2) for n, a in zip(d.N, d["mean"])},
                             std={int(n): round(sd * 100, 2) for n, sd in zip(d.N, d["std"].fillna(0))},
                             time={int(n): round(t, 3) for n, t in zip(d.N, d.time)})
    summary["e1"] = s


def fig_e2():
    df = load("e2_length.csv")
    if df is None:
        return
    g = agg(df, ["n", "N"])
    rows = []
    for n, d in g.groupby("n"):
        d = d.sort_values("N")
        rows.append(dict(n=n, N95=crps_to_reach(d.N, d["mean"], 0.95), N99=crps_to_reach(d.N, d["mean"], 0.99)))
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for col, c, mk, lab in [("N95", BLUE_RAMP[1], "o", "95 % accuracy"), ("N99", BLUE_RAMP[4], "s", "99 % accuracy")]:
        ax.plot(r.n, r[col], color=c, marker=mk, label=lab)
        slope = np.polyfit(r.n, r[col], 1)[0]
        ax.text(r.n.iloc[-1] * 1.03, r[col].iloc[-1], f"≈ {slope:.0f}·n", color=INK2, fontsize=8.5, va="center")
    ax.set_xlabel("Challenge length n (number of arbiter stages)")
    ax.set_ylabel("CRPs needed (logistic regression)")
    ax.set_title("Attack cost grows only linearly with n", loc="left")
    ax.set_xlim(0, 300); ax.legend(loc="upper left")
    fig.savefig(os.path.join(FIG, "fig_e2_length.png")); plt.close(fig)
    summary["e2"] = r.round(1).to_dict(orient="records")


def fig_e3():
    df = load("e3_xor.csv")
    if df is None:
        return
    df["features"] = "phi"
    g = agg(df, ["k", "model", "features", "N"])
    ks = sorted(g.k.unique())
    ncol = 2 if len(ks) > 2 else len(ks)
    nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, k) in enumerate(zip(axes, ks)):
        plot_model_curves(ax, g[g.k == k], ["LR", "RBFSVM", "MLP", "PLR"])
        finish_acc_axis(ax, f"{k}-XOR Arbiter PUF")
        Ns = sorted(g[g.k == k].N.unique())
        ax.set_xticks(Ns, [f"{n / 1000:g}k" for n in Ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("Accuracy on unseen challenges (%)" if i % ncol == 0 else "")
        ax.set_xlabel("Training CRPs (log scale)")
    for ax in axes[len(ks):]:
        ax.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_e3_xor.png")); plt.close(fig)

    # how N95 grows with k
    rows = []
    for (k, m), d in g.groupby(["k", "model"]):
        d = d.sort_values("N")
        rows.append(dict(k=int(k), model=m, N95=crps_to_reach(d.N, d["mean"], 0.95),
                         best_acc=round(d["mean"].max() * 100, 2), N_at_best=int(d.N.iloc[d["mean"].values.argmax()]),
                         time_at_best=round(float(d.time.iloc[d["mean"].values.argmax()]), 1)))
    r = pd.DataFrame(rows)
    summary["e3"] = r.to_dict(orient="records")
    summary["e3_curves"] = {f"{m}_k{k}": {int(n): round(a * 100, 2) for n, a in zip(d.N, d["mean"])}
                            for (k, m), d in g.groupby(["k", "model"])}
    summary["e3_times"] = {f"{m}_k{k}": {int(n): round(t, 1) for n, t in zip(d.N, d.time)}
                           for (k, m), d in g.groupby(["k", "model"])}
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    fits = {}
    for m in ["MLP", "PLR"]:
        d = r[(r.model == m) & r.N95.notna()].sort_values("k")
        if len(d) < 2:
            continue
        c, mk, lab = MODEL_STYLE[m]
        ax.plot(d.k, d.N95, color=c, marker=mk, label=lab, ms=7)
        b, a = np.polyfit(d.k, np.log10(d.N95), 1)
        fits[m] = dict(log10_slope=b, intercept=a, factor_per_k=10 ** b)
        kx = np.arange(d.k.min(), 9)
        ax.plot(kx, 10 ** (a + b * kx), color=c, ls="--", lw=1, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Number of XORed arbiter chains k")
    ax.set_ylabel("CRPs needed for 95 % accuracy (log)")
    ax.set_title("Each extra XOR multiplies the attacker's cost", loc="left")
    ax.text(6.05, ax.get_ylim()[0] * 3, "dashed = log-linear\nextrapolation", color=INK2, fontsize=8)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(FIG, "fig_e3_scaling.png")); plt.close(fig)
    summary["e3_fits"] = fits


def fig_e4():
    df = load("e4_noise.csv")
    if df is None:
        return
    df["reliability"] = df["reliability"].fillna(1.0)
    g = df.groupby(["k", "model", "noisiness", "N"]).agg(acc=("acc", "mean"), acc_noisy=("acc_vs_noisy", "mean"),
                                                         rel=("reliability", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, k, m in [(axes[0], 1, "LR"), (axes[1], 2, "MLP")]:
        d = g[(g.k == k) & (g.model == m)]
        for i, (N, dd) in enumerate(d.groupby("N")):
            dd = dd.sort_values("noisiness")
            c = BLUE_RAMP[min(len(BLUE_RAMP) - 1, i + (1 if k == 2 else 0))]
            ax.plot(dd.rel * 100, dd.acc * 100, color=c, marker="o", label=f"N = {N:,}")
        rel = d.groupby("noisiness").rel.mean().sort_index()
        ax.plot(rel * 100, rel * 100, color=INK2, ls=":", lw=1, label="chip's own reliability")
        ax.invert_xaxis()
        ax.set_xlabel("PUF reliability (% identical re-evaluations)")
        ax.set_title(f"{'Arbiter' if k == 1 else '2-XOR'} PUF, {MODEL_STYLE[m][2]}", loc="left")
        ax.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel("Accuracy vs. noise-free response (%)")
    axes[0].set_ylim(80, 100.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e4_noise.png")); plt.close(fig)
    summary["e4"] = g.round(4).to_dict(orient="records")


def fig_e5():
    df = load("e5_weights.csv")
    if df is None:
        return
    Ns = sorted(df.N.unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.1 * len(Ns), 3.2))
    for ax, N in zip(axes, Ns):
        d = df[df.N == N]
        wt, wh = d.w_true.values, d.w_hat.values
        wh = wh * (wt @ wh) / (wh @ wh)  # LR only learns w up to scale
        ax.scatter(wt, wh, s=22, color=MODEL_STYLE["LR"][0], edgecolor="white", lw=0.6)
        lim = np.abs(np.r_[wt, wh]).max() * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls=":")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"N = {N:,} CRPs   r = {d["corr"].iloc[0]:.3f}", loc="left", fontsize=10)
        ax.set_xlabel(r"True stage delay parameter $w_i$")
    axes[0].set_ylabel("Parameter learned by LR (scaled)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e5_weights.png")); plt.close(fig)
    summary["e5"] = {int(N): round(float(df[df.N == N]["corr"].iloc[0]), 4) for N in Ns}


def e0_table():
    df = load("e0_quality.csv")
    if df is not None:
        summary["e0"] = df.round(4).to_dict(orient="records")


if __name__ == "__main__":
    for f in (e0_table, fig_e1, fig_e2, fig_e3, fig_e4, fig_e5):
        f()
    rows = [dict(k=r["k"], model=r["model"], N95=r["N95"], best_acc=r["best_acc"]) for r in summary.get("e3", [])]
    for m in ("LR_phi", "MLP_phi"):
        if m in summary.get("e1", {}):
            rows.insert(0, dict(k=1, model=m.split("_")[0], N95=summary["e1"][m]["N95"],
                                best_acc=max(summary["e1"][m]["acc"].values())))
    if rows:
        t = pd.DataFrame(rows).sort_values(["k", "model"])
        t["N95"] = t["N95"].round(-1)
        t.to_csv(os.path.join(RES, "n95_summary.csv"), index=False)
        print(t.to_string(index=False))
    print("figures:", sorted(os.listdir(FIG)))
