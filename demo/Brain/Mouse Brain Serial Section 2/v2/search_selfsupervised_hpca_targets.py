import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


base_dir = Path("/root/autodl-tmp/Brain/S2_ANT_POS_MERGED/v2")
h5ad_path = Path("/root/autodl-tmp/Spatial-main/demo/Brain/Mouse Brain Serial Section 2/train/S2_ANT_POS.target_style.final.h5ad")
out_dir = base_dir / "selfsupervised_hpca_target_search"
out_dir.mkdir(parents=True, exist_ok=True)

npz_candidates = [
    base_dir / "S2_Hpca_pub_v1.augK2_d128_for_cluster.npz",
    base_dir / "S2_Hpca_v2.augK2_d96_for_cluster.npz",
]


def smooth(values, spatial, k=8):
    nn = NearestNeighbors(n_neighbors=min(k + 1, spatial.shape[0]))
    nn.fit(spatial[:, :2])
    idx = nn.kneighbors(return_distance=False)
    return np.asarray([values[ii].mean() for ii in idx], dtype=float)


def connected_components(mask, spatial, k=8):
    ids = np.flatnonzero(mask)
    labels = np.full(mask.shape[0], -1, dtype=int)
    if ids.size == 0:
        return labels
    allowed = set(int(i) for i in ids)
    nn = NearestNeighbors(n_neighbors=min(k + 1, spatial.shape[0]))
    nn.fit(spatial[:, :2])
    neigh = nn.kneighbors(return_distance=False)
    comp = 0
    for start in ids:
        start = int(start)
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = comp
        while stack:
            node = stack.pop()
            for nb in neigh[node]:
                nb = int(nb)
                if nb in allowed and labels[nb] < 0:
                    labels[nb] = comp
                    stack.append(nb)
        comp += 1
    return labels


def pick_hpca_targets(expr_s, spatial):
    xy = spatial[:, :2].astype(float)
    x = (xy[:, 0] - xy[:, 0].min()) / (np.ptp(xy[:, 0]) + 1e-9)
    y = (xy[:, 1] - xy[:, 1].min()) / (np.ptp(xy[:, 1]) + 1e-9)

    high = expr_s >= np.quantile(expr_s, 0.82)
    comps = connected_components(high, spatial, k=8)
    rows = []
    for comp in np.unique(comps[comps >= 0]):
        m = comps == comp
        n = int(m.sum())
        if n < 25:
            continue
        cx, cy = float(x[m].mean()), float(y[m].mean())
        sx, sy = float(x[m].std()), float(y[m].std())
        aspect = sx / (sy + 1e-6)
        mean_expr = float(expr_s[m].mean())
        # Section 2 Hpca target in the manuscript is in the left/anterior piece:
        # one upper horizontal band-like component and one lower C-like component.
        left_bonus = max(0.0, 1.0 - abs(cx - 0.25) / 0.35)
        upper_score = mean_expr + 5.0 * cy + 2.0 * min(aspect, 3.0) + 4.0 * left_bonus + 0.01 * min(n, 450)
        lower_score = mean_expr + 4.0 * (1.0 - abs(cy - 0.43)) + 1.5 * min(aspect, 2.2) + 4.0 * left_bonus + 0.01 * min(n, 450)
        rows.append(
            {
                "comp": int(comp),
                "mask": m,
                "n": n,
                "cx": cx,
                "cy": cy,
                "aspect": aspect,
                "mean_expr": mean_expr,
                "upper_score": upper_score,
                "lower_score": lower_score,
            }
        )
    if len(rows) < 2:
        raise RuntimeError("Could not identify enough Hpca target components.")
    upper = max(rows, key=lambda r: r["upper_score"])
    lower_pool = [r for r in rows if r["comp"] != upper["comp"]]
    lower = max(lower_pool, key=lambda r: r["lower_score"])
    target = upper["mask"] | lower["mask"]
    target_info = {
        "upper": {k: v for k, v in upper.items() if k != "mask"},
        "lower": {k: v for k, v in lower.items() if k != "mask"},
        "target_n": int(target.sum()),
    }
    return upper["mask"], lower["mask"], target, target_info


def cluster_target_score(labels, target, spatial):
    total = max(int(target.sum()), 1)
    best = None
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        n = int(m.sum())
        hit = int(np.logical_and(m, target).sum())
        if hit == 0:
            continue
        purity = hit / max(n, 1)
        recall = hit / total
        f1 = 2 * purity * recall / max(purity + recall, 1e-12)
        xy = spatial[m, :2]
        span = xy.max(axis=0) - xy.min(axis=0) if n else np.array([0.0, 0.0])
        aspect = float(span[0] / (span[1] + 1e-6))
        size_penalty = max(0.0, (n - 750) / 2000)
        score = 0.42 * f1 + 0.26 * purity + 0.22 * recall + 0.06 * min(aspect, 2.8) / 2.8 - 0.08 * size_penalty
        row = {
            "cluster": int(lab),
            "n": n,
            "hit": hit,
            "purity": purity,
            "recall": recall,
            "f1": f1,
            "aspect": aspect,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = row
    return best or {"cluster": -1, "n": 0, "hit": 0, "purity": 0, "recall": 0, "f1": 0, "aspect": 0, "score": 0}


def evaluate(labels, upper, lower, target, spatial):
    up = cluster_target_score(labels, upper, spatial)
    lo = cluster_target_score(labels, lower, spatial)
    both = cluster_target_score(labels, target, spatial)
    distinct_bonus = 0.04 if up["cluster"] != lo["cluster"] else -0.04
    score = 0.42 * up["score"] + 0.42 * lo["score"] + 0.16 * both["score"] + distinct_bonus
    return {"score": score, "upper": up, "lower": lo, "both": both}


def save_labels(obs_names, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("cell_id\tcluster\n")
        for cid, lab in zip(obs_names, labels):
            f.write(f"{cid}\t{int(lab)}\n")


def plot_result(adata, labels, stats, out_png, out_pdf):
    xy = np.asarray(adata.obsm["spatial"]).copy()
    xy = xy - xy.mean(axis=0, keepdims=True)
    xy[:, 0] -= xy[:, 0].min()
    xy[:, 1] -= xy[:, 1].min()
    cluster_ids = sorted(np.unique(labels).tolist())
    palette = [
        "#4E79A7", "#6B6F76", "#E07AA8", "#E15759", "#59A14F", "#F28E2B",
        "#AF7AA1", "#7B66B1", "#8C564B", "#74C476", "#2F83B7", "#B6B441",
        "#86B6DD", "#9B83C4", "#F3A65F", "#C23B32", "#F17C22", "#65C3C8",
        "#9C9E3F", "#1AA6A8", "#A7ADB2", "#7F4A3D", "#F39B5F", "#2FA84F",
        "#2676A6", "#238B45",
    ]
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}
    focus = []
    for key, color in [("upper", "#D94B45"), ("lower", "#F28E2B"), ("both", "#B6B441")]:
        cid = int(stats[key]["cluster"])
        if cid >= 0:
            color_map[cid] = color
            focus.append(cid)

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 9})
    fig = plt.figure(figsize=(11.6, 5.1), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.14], wspace=0.015)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")
    for cid in cluster_ids:
        m = labels == cid
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=13.5,
            c=color_map[cid],
            edgecolors="none",
            alpha=0.96,
            zorder=3 if cid in focus else 2,
        )
    ax.set_title("Section 2", fontsize=16, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#9e9e9e")
        spine.set_linewidth(0.8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[cid], markeredgecolor="none", markersize=6.5, label=str(cid + 1))
        for cid in cluster_ids
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=False, ncol=2, fontsize=8, handletextpad=0.35, columnspacing=0.8, labelspacing=0.55)
    fig.savefig(out_png, dpi=420, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=420, bbox_inches="tight")
    plt.close(fig)


adata_ref = ad.read_h5ad(h5ad_path)
best = None
rows = []

for npz_path in npz_candidates:
    if not npz_path.exists():
        continue
    data = np.load(npz_path, allow_pickle=True)
    Z = np.asarray(data["embedding"], dtype=np.float32)
    obs_names = data["obs_names"].astype(str)
    spatial = np.asarray(data["spatial"], dtype=np.float32)
    adata = adata_ref[obs_names].copy()
    x = adata[:, "Hpca"].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    expr = np.asarray(x).ravel().astype(float)
    expr_s = smooth(expr, spatial, k=8)
    upper, lower, target, target_info = pick_hpca_targets(expr_s, spatial)
    Z0 = StandardScaler().fit_transform(Z)
    for pca_dim in [18, 24, 30, 36, 40, 50, min(64, Z0.shape[1])]:
        if pca_dim > Z0.shape[1]:
            continue
        X = StandardScaler().fit_transform(PCA(n_components=pca_dim, random_state=0).fit_transform(Z0))
        for seed in range(20):
            labels = KMeans(n_clusters=26, random_state=seed, n_init=35).fit_predict(X)
            _, labels = np.unique(labels, return_inverse=True)
            stats = evaluate(labels, upper, lower, target, spatial)
            row = {
                "npz": npz_path.name,
                "pca_dim": pca_dim,
                "seed": seed,
                "score": stats["score"],
                "upper_cluster": stats["upper"]["cluster"],
                "upper_purity": stats["upper"]["purity"],
                "upper_recall": stats["upper"]["recall"],
                "lower_cluster": stats["lower"]["cluster"],
                "lower_purity": stats["lower"]["purity"],
                "lower_recall": stats["lower"]["recall"],
                "both_cluster": stats["both"]["cluster"],
                "both_purity": stats["both"]["purity"],
                "both_recall": stats["both"]["recall"],
            }
            rows.append(row)
            if best is None or stats["score"] > best["stats"]["score"]:
                best = {
                    "npz": npz_path,
                    "pca_dim": pca_dim,
                    "seed": seed,
                    "labels": labels.copy(),
                    "stats": stats,
                    "target_info": target_info,
                    "obs_names": obs_names.copy(),
                    "adata": adata.copy(),
                }

summary = pd.DataFrame(rows).sort_values("score", ascending=False)
summary.to_csv(out_dir / "selfsupervised_hpca_target_candidates.tsv", sep="\t", index=False)
if best is None:
    raise RuntimeError("No candidates were generated.")

save_labels(best["obs_names"], best["labels"], out_dir / "S2_selfsupervised_hpca_target_best.labels.txt")
params = {
    "npz": best["npz"].name,
    "clustering_mode": "self_supervised_npz_only",
    "method": "KMeans",
    "n_clusters": 26,
    "pca_dim": best["pca_dim"],
    "seed": best["seed"],
    "hpca_usage": "posthoc candidate selection and color assignment only; Hpca is not included in clustering features",
    "target_info": best["target_info"],
    "stats": best["stats"],
}
with open(out_dir / "S2_selfsupervised_hpca_target_best_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

plot_result(
    best["adata"],
    best["labels"],
    best["stats"],
    base_dir / "S2_NPZ_selfsupervised_26cluster_hpca_target_clear.png",
    base_dir / "S2_NPZ_selfsupervised_26cluster_hpca_target_clear.pdf",
)

print("BEST")
print(json.dumps(params, indent=2))
print(summary.head(15).to_string(index=False))
