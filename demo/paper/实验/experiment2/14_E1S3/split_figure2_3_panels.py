from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


ROOT = Path(r"D:\Spatial-main")
DATA_PATH = ROOT / "dataset" / "Mouse embryo" / "E14.5_E1S3.MOSTA.h5ad"
LABEL_PATH = (
    ROOT
    / "demo"
    / "mouse"
    / "14_E1S3"
    / "train"
    / "E14.5_E1S3_SSL_A_leiden.leiden.labels.txt"
)
OUT_ROOT = (
    ROOT
    / "demo"
    / "\u5b9e\u9a8c"
    / "experiment2"
    / "14_E1S3"
    / "single_panels"
)
OUT_TISSUE = OUT_ROOT / "2_tissue"
OUT_GENE = OUT_ROOT / "3_gene"

TISSUES = [
    "Heart",
    "Liver",
    "Olfactory epithelium",
    "Cartilage primordium",
    "Muscle",
    "Epidermis",
    "Meninges",
    "Brain",
]

TISSUE_COLORS = {
    "Heart": "#d63b75",
    "Liver": "#9a4cc2",
    "Olfactory epithelium": "#5aa0e6",
    "Cartilage primordium": "#57c56b",
    "Muscle": "#c91f5a",
    "Epidermis": "#3f7fd2",
    "Meninges": "#d9c45f",
    "Brain": "#ea8737",
}

CLUSTER_TO_TISSUE = {
    "1": "Epidermis",
    "2": "Muscle",
    "3": "Meninges",
    "4": "Brain",
    "5": "Liver",
    "6": "Brain",
    "7": "Cartilage primordium",
    "9": "Heart",
    "11": "Cartilage primordium",
    "13": "Olfactory epithelium",
    "14": "Brain",
}

GENES = ["Col2a1", "Krt5", "Myh3", "Fabp7", "Dbi", "Mt3", "Stmn2", "Tubb3", "Gap43"]

BG_COLOR = "#cfcfd4"
TISSUE_BG_SIZE = 0.9
TISSUE_FG_SIZE = 0.9
GENE_SIZE = 0.68


def file_stem(name):
    return name.lower().replace(" ", "_")


def load_pred_labels(obs_names):
    labels = pd.read_csv(LABEL_PATH, sep="\t").set_index("cell_id")
    common = pd.Index(obs_names).intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No overlapping cell_id values between h5ad and labels.")
    return labels.loc[common, "cluster"].astype(str), common


def spatial_limits(x, y, pad=10):
    return x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad


def clean_axis(ax, xlim, ylim):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_tissue_panel(x, y, mask, title, color, out_path):
    fig, ax = plt.subplots(figsize=(2.15, 2.45), dpi=300)
    xlim = (x.min() - 10, x.max() + 10)
    ylim = (y.min() - 10, y.max() + 10)

    ax.scatter(
        x[~mask],
        y[~mask],
        s=TISSUE_BG_SIZE,
        c=BG_COLOR,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        x[mask],
        y[mask],
        s=TISSUE_FG_SIZE,
        c=color,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=13, pad=6)
    clean_axis(ax, xlim, ylim)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def get_expr(adata, gene):
    expr = adata[:, gene].X
    if sparse.issparse(expr):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()
    return expr.astype(float)


def process_expr(expr, nn_idx):
    expr = np.asarray(expr, dtype=float).copy()
    expr[expr < 0] = 0.0
    expr = np.log1p(expr)
    low = np.percentile(expr, 3.0)
    high = np.percentile(expr, 99.5)
    expr[expr < low] = 0.0
    expr = np.clip(expr, 0.0, high)
    expr = 0.88 * expr + 0.12 * expr[nn_idx].mean(axis=1)
    vmax = np.percentile(expr, 99.5)
    if vmax <= 0:
        vmax = 1.0
    return expr + 1e-8, vmax


def draw_gene_panel(x, y, expr, vmax, title, out_path):
    fig, ax = plt.subplots(figsize=(2.0, 2.72), dpi=300)
    xlim = (x.min() - 10, x.max() + 10)
    ylim = (y.min() - 10, y.max() + 10)
    ax.scatter(
        x,
        y,
        c=expr,
        s=GENE_SIZE,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        alpha=0.95,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=15, pad=5)
    clean_axis(ax, xlim, ylim)
    fig.savefig(out_path, dpi=700, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    OUT_TISSUE.mkdir(parents=True, exist_ok=True)
    OUT_GENE.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    pred_cluster, common = load_pred_labels(adata.obs_names)
    adata = adata[common].copy()
    adata.obs["pred_cluster"] = pred_cluster.loc[adata.obs_names].values
    adata.obs["pred_tissue"] = (
        adata.obs["pred_cluster"].astype(str).map(CLUSTER_TO_TISSUE).fillna("Other")
    )

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    x = coords[:, 0]
    y = -coords[:, 1]

    for tissue in TISSUES:
        stem = file_stem(tissue)
        color = TISSUE_COLORS[tissue]

        gt_mask = adata.obs["annotation"].astype(str).values == tissue
        draw_tissue_panel(x, y, gt_mask, tissue, color, OUT_TISSUE / f"{stem}_g.png")

        pred_mask = adata.obs["pred_tissue"].astype(str).values == tissue
        draw_tissue_panel(x, y, pred_mask, tissue, color, OUT_TISSUE / f"{stem}_o.png")

    missing = [gene for gene in GENES if gene not in adata.var_names]
    if missing:
        raise ValueError(f"Missing genes: {missing}")

    nbrs = NearestNeighbors(n_neighbors=9, algorithm="auto").fit(coords)
    _, nn_idx = nbrs.kneighbors(coords)
    nn_idx = nn_idx[:, 1:]

    for gene in GENES:
        expr, vmax = process_expr(get_expr(adata, gene), nn_idx)
        draw_gene_panel(x, y, expr, vmax, gene, OUT_GENE / f"{gene}.png")

    print(f"Saved tissue panels: {OUT_TISSUE}")
    print(f"Saved gene panels: {OUT_GENE}")


if __name__ == "__main__":
    main()
