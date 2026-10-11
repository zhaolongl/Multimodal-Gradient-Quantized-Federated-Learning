"""Aggregate repeated 5x5 CV results: summary table, corrected CIs, paired differences, Fig. S2."""
import json, glob, os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else "cv_original"
recs = [json.load(open(f, encoding="utf8")) for f in sorted(glob.glob(os.path.join(HERE, src, "partition_*.json")))]
J = len(recs)
print("partitions:", J)
rows = []
for r in recs:
    for x in r["results"]:
        rows.append(dict(partition=r["partition"], repeat=r["repeat"], fold=r["fold"], n_train=r["n_train"],
                         n_test=r["n_test"], model=x["model"], regime=x["regime"], accuracy=x["accuracy"],
                         macro_f1=x["macro_f1"], best_round_accuracy=x.get("best_round_accuracy", np.nan),
                         last10=x.get("last10_mean_accuracy", np.nan), comm_MB=x.get("comm_MB", np.nan)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, f"{src}_per_partition.csv"), index=False)
ratio = (df.n_test / df.n_train).mean()


def corrected_ci(v):
    v = np.asarray(v, float)
    m, s, k = v.mean(), v.std(ddof=1), len(v)
    se = np.sqrt((1 / k + ratio) * s ** 2)
    t = stats.t.ppf(0.975, k - 1)
    return m, s, max(0.0, m - t * se), min(1.0, m + t * se)


order_m = ["MultiModalMLP", "EarlyFusionMLP", "MultiModalCNN1D", "RandomForest"]
order_r = ["Centralized", "Fed", "Fed-Comp"]
single = {("MultiModalMLP", "Centralized"): 0.96, ("MultiModalMLP", "Fed"): 0.93, ("MultiModalMLP", "Fed-Comp"): 0.93,
          ("EarlyFusionMLP", "Centralized"): 0.72, ("EarlyFusionMLP", "Fed"): 0.46, ("EarlyFusionMLP", "Fed-Comp"): 0.50,
          ("MultiModalCNN1D", "Centralized"): 0.68, ("MultiModalCNN1D", "Fed"): 0.82, ("MultiModalCNN1D", "Fed-Comp"): 0.64}
summ = []
for m in order_m:
    for g in order_r:
        sub = df[(df.model == m) & (df.regime == g)]
        if sub.empty:
            continue
        am, asd, alo, ahi = corrected_ci(sub.accuracy)
        fm, fsd, flo, fhi = corrected_ci(sub.macro_f1)
        # pooled out-of-fold accuracy per repeat
        pooled = [np.average(s.accuracy, weights=s.n_test) for _, s in sub.groupby("repeat")]
        summ.append(dict(model=m, regime=g, n=len(sub), acc_mean=am, acc_sd=asd, acc_ci_lo=alo, acc_ci_hi=ahi,
                         acc_min=sub.accuracy.min(), acc_max=sub.accuracy.max(), acc_median=sub.accuracy.median(),
                         f1_mean=fm, f1_sd=fsd, f1_ci_lo=flo, f1_ci_hi=fhi,
                         pooled_repeat_min=min(pooled), pooled_repeat_max=max(pooled),
                         best_round_mean=sub.best_round_accuracy.mean(), last10_mean=sub.last10.mean(),
                         comm_MB=sub.comm_MB.mean(), single_split=single.get((m, g), np.nan)))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(HERE, f"{src}_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(S.round(3).to_string())

# paired differences (same partitions)
piv = df.pivot_table(index="partition", columns=["model", "regime"], values="accuracy")
diffs = {}
def pdiff(a, b, label):
    v = (piv[a] - piv[b]).dropna().values
    m, s, lo, hi = v.mean(), v.std(ddof=1), None, None
    se = np.sqrt((1 / len(v) + ratio) * s ** 2)
    t = stats.t.ppf(0.975, len(v) - 1)
    diffs[label] = dict(mean=m, sd=s, lo=m - t * se, hi=m + t * se, n=len(v))
    print(f"{label:55s} {m:+.3f} (95% CI {m - t * se:+.3f} to {m + t * se:+.3f})")
for m in ["MultiModalMLP", "EarlyFusionMLP", "MultiModalCNN1D"]:
    if (m, "Fed") in piv:
        pdiff((m, "Fed"), (m, "Centralized"), f"{m}: Fed - Centralized")
        pdiff((m, "Fed-Comp"), (m, "Fed"), f"{m}: Fed-Comp - Fed")
        pdiff((m, "Fed-Comp"), (m, "Centralized"), f"{m}: Fed-Comp - Centralized")
if ("RandomForest", "Centralized") in piv and ("MultiModalMLP", "Centralized") in piv:
    pdiff(("MultiModalMLP", "Centralized"), ("RandomForest", "Centralized"), "MultiModalMLP Centralized - RF")
    pdiff(("MultiModalMLP", "Fed-Comp"), ("RandomForest", "Centralized"), "MultiModalMLP Fed-Comp - RF")
for m in ["MultiModalMLP", "EarlyFusionMLP", "MultiModalCNN1D"]:
    for g in ["Fed", "Fed-Comp"]:
        sub = df[(df.model == m) & (df.regime == g)]
        if not sub.empty:
            print(f"{m} {g}: best-round minus final-round = {(sub.best_round_accuracy - sub.accuracy).mean():+.3f}")
json.dump(dict(J=J, ratio=ratio, summary=summ, diffs=diffs), open(os.path.join(HERE, f"{src}_stats.json"), "w"), indent=1, default=float)

# ---------- Fig. S2 ----------
if src == "cv_original":
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    cols = {"Centralized": "#0072B2", "Fed": "#E69F00", "Fed-Comp": "#009E73"}
    labels = {"MultiModalMLP": "MultiModalMLP", "EarlyFusionMLP": "EarlyFusionMLP",
              "MultiModalCNN1D": "MultiModalCNN1D", "RandomForest": "RF baseline"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["(A) Accuracy", "(B) Macro-averaged F1 score"]):
        for i, m in enumerate(order_m):
            regs = order_r if m != "RandomForest" else ["Centralized"]
            for j, g in enumerate(regs):
                x0 = i + (j - (len(regs) - 1) / 2) * 0.26
                v = df[(df.model == m) & (df.regime == g)][metric].values
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[x0], widths=0.2, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.7),
                                capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
                bp["boxes"][0].set_facecolor(cols[g] if m != "RandomForest" else "#999999")
                bp["boxes"][0].set_alpha(0.35)
                ax.scatter(x0 + rng.uniform(-0.06, 0.06, len(v)), v, s=6, color=cols[g] if m != "RandomForest" else "#555555",
                           alpha=0.8, lw=0, zorder=3)
                ax.scatter([x0], [v.mean()], marker="D", s=14, facecolor="white", edgecolor="black", lw=0.7, zorder=4)
                if metric == "accuracy" and (m, g) in single:
                    ax.scatter([x0], [single[(m, g)]], marker="*", s=40, color="#D55E00", lw=0, zorder=5)
        ax.set_xticks(range(len(order_m)))
        ax.set_xticklabels([labels[m] for m in order_m], rotation=15)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test-fold value (25 partitions)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(facecolor=cols[g], alpha=0.5, label=g) for g in order_r] + \
        [Patch(facecolor="#999999", alpha=0.5, label="RF (centralized)"),
         Line2D([], [], marker="D", ls="", mfc="white", mec="black", label="Mean"),
         Line2D([], [], marker="*", ls="", color="#D55E00", ms=8, label="Single 80:20 split (Fig. 6D)")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"Fig_S2.{ext}"), dpi=600 if ext == "png" else None)
    print("Fig. S2 saved")
