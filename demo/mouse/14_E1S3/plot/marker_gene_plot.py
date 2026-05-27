import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

h5ad_path = "/root/autodl-tmp/Mouse/E14.5_E1S3.MOSTA.h5ad"
out_png = "/root/autodl-tmp/paper_figures/E14.5_E1S3_marker_genes_better_than_D.png"

marker_genes = [
    "Col2a1", "Krt5", "Myh3", "Fabp7", "Dbi",
    "Mt3", "Stmn2", "Tubb3", "Gap43"
]

flip_y = True
dot_size = 0.68
alpha = 0.95
cmap = "viridis"
title_size = 15
pad = 10
low_clip_percentile = 3.0
high_clip_percentile = 99.5
smooth_k = 8
smooth_weight = 0.12
expr_eps = 1e-8

adata = sc.read_h5ad(h5ad_path)

if "spatial" not in adata.obsm:
    raise ValueError("adata.obsm 中没有 'spatial'")

present_genes = [g for g in marker_genes if g in adata.var_names]
missing_genes = [g for g in marker_genes if g not in adata.var_names]

if len(present_genes) == 0:
    raise ValueError(f"这些基因都不在 adata.var_names 中: {marker_genes}")

if missing_genes:
    print("[WARN] 跳过缺失基因:", missing_genes)

coords = np.asarray(adata.obsm["spatial"]).astype(float)
x = coords[:, 0]
y_raw = coords[:, 1]
y = -y_raw if flip_y else y_raw

xmin, xmax = x.min() - pad, x.max() + pad
ymin, ymax = y.min() - pad, y.max() + pad

nbrs = NearestNeighbors(n_neighbors=smooth_k + 1, algorithm="auto").fit(coords)
_, nn_idx = nbrs.kneighbors(coords)
nn_idx = nn_idx[:, 1:]

def get_expr(gene):
    expr = adata[:, gene].X
    if sparse.issparse(expr):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()
    return expr.astype(float)

def robust_process(expr):
    expr = np.asarray(expr, dtype=float).copy()
    expr[expr < 0] = 0.0

    expr = np.log1p(expr)

    low_thr = np.percentile(expr, low_clip_percentile)
    high_thr = np.percentile(expr, high_clip_percentile)

    expr[expr < low_thr] = 0.0
    expr = np.clip(expr, 0.0, high_thr)

    neigh_mean = expr[nn_idx].mean(axis=1)
    expr = (1.0 - smooth_weight) * expr + smooth_weight * neigh_mean

    vmax = np.percentile(expr, 99.5)
    if vmax <= 0:
        vmax = 1.0

    return expr + expr_eps, vmax

n = len(present_genes)

fig, axes = plt.subplots(
    1,
    n + 1,
    figsize=(2.0 * n + 0.38, 2.72),
    dpi=300,
    gridspec_kw={"width_ratios": [1] * n + [0.15]}
)

last_sc = None

for i, gene in enumerate(present_genes):
    ax = axes[i]

    expr = get_expr(gene)
    expr_plot, vmax = robust_process(expr)

    last_sc = ax.scatter(
        x,
        y,
        c=expr_plot,
        s=dot_size,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=alpha,
        edgecolors="none",
        linewidths=0,
        rasterized=True
    )

    ax.set_title(gene, fontsize=title_size, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)

cax = axes[-1]
cb = plt.colorbar(last_sc, cax=cax)
cb.set_ticks([])
cb.outline.set_visible(False)

cax.set_title("Expression", fontsize=11, pad=6)
cax.text(1.05, 0.98, "High", transform=cax.transAxes, ha="left", va="top", fontsize=10)
cax.text(1.05, 0.02, "Low", transform=cax.transAxes, ha="left", va="bottom", fontsize=10)

plt.tight_layout(pad=0.45)
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=700, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved: {out_png}")