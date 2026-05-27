import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. 输入输出
# =========================================================
h5ad_path = "/root/autodl-tmp/Mouse/E14.5_E1S3.MOSTA.h5ad"
pred_label_path = "/root/autodl-tmp/Spatial-main/demo/mouse/14_E1S3/E14.5_E1S3_SSL_A_leiden.leiden.labels.txt"

out_dir = Path("/root/autodl-tmp/paper_figures")
out_dir.mkdir(parents=True, exist_ok=True)

out_png = out_dir / "E14.5_E1S3_tissue_panels_final_manual.png"
out_pdf = out_dir / "E14.5_E1S3_tissue_panels_final_manual.pdf"
out_csv = out_dir / "E14.5_E1S3_pred_matched_labels.csv"

# =========================================================
# 2. 读取数据
# =========================================================
adata = sc.read_h5ad(h5ad_path)
pred = pd.read_csv(pred_label_path, sep="\t").set_index("cell_id")

common = adata.obs_names.intersection(pred.index)
adata = adata[common].copy()
pred = pred.loc[common].copy()

adata.obs["pred_cluster"] = pred["cluster"].astype(str)

gt_col = "annotation"
adata.obs["gt_label"] = adata.obs[gt_col].astype(str)

# =========================================================
# 3. 最终人工映射
# =========================================================
cluster_to_tissue = {
    "1":  "Epidermis",
    "2":  "Muscle",
    "3":  "Meninges",
    "4":  "Brain",
    "5":  "Liver",
    "6":  "Brain",
    "7":  "Cartilage primordium",
    "9":  "Heart",
    "11": "Cartilage primordium",
    "13": "Olfactory epithelium",
    "14": "Brain",
}

tissue_to_id = {
    "Heart": 0,
    "Liver": 1,
    "Olfactory epithelium": 2,
    "Cartilage primordium": 3,
    "Muscle": 4,
    "Epidermis": 5,
    "Meninges": 6,
    "Brain": 7,
    "Other": -1
}

tissue_colors = {
    "Heart": "#d63b75",
    "Liver": "#9a4cc2",
    "Olfactory epithelium": "#5aa0e6",
    "Cartilage primordium": "#57c56b",
    "Muscle": "#c91f5a",
    "Epidermis": "#3f7fd2",
    "Meninges": "#d9c45f",
    "Brain": "#ea8737",
}
bg_color = "#cfcfd4"

tissues = [
    "Heart",
    "Liver",
    "Olfactory epithelium",
    "Cartilage primordium",
    "Muscle",
    "Epidermis",
    "Meninges",
    "Brain",
]

# =========================================================
# 4. 写入映射结果
# =========================================================
adata.obs["pred_matched_tissue"] = adata.obs["pred_cluster"].map(cluster_to_tissue)
adata.obs["pred_matched_tissue"] = adata.obs["pred_matched_tissue"].fillna("Other")
adata.obs["pred_matched_id"] = adata.obs["pred_matched_tissue"].map(tissue_to_id)

adata.obs[["pred_cluster", "pred_matched_tissue", "pred_matched_id"]].to_csv(out_csv)

# =========================================================
# 5. 空间坐标
# =========================================================
x = adata.obsm["spatial"][:, 0]
y = -adata.obsm["spatial"][:, 1]

pad = 10
xmin, xmax = x.min() - pad, x.max() + pad
ymin, ymax = y.min() - pad, y.max() + pad

# =========================================================
# 6. 画 one-vs-rest 面板图
# =========================================================
fig, axes = plt.subplots(
    2, len(tissues),
    figsize=(2.15 * len(tissues), 4.5),
    dpi=300
)

for j, tissue in enumerate(tissues):
    color = tissue_colors[tissue]

    # ground truth
    ax = axes[0, j]
    gt_mask = (adata.obs["gt_label"].values == tissue)

    ax.scatter(
        x[~gt_mask], y[~gt_mask],
        s=0.9, c=bg_color, edgecolors="none", linewidths=0, rasterized=True
    )
    ax.scatter(
        x[gt_mask], y[gt_mask],
        s=0.9, c=color, edgecolors="none", linewidths=0, rasterized=True
    )

    ax.set_title(tissue, fontsize=13, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # ours
    ax = axes[1, j]
    pred_mask = (adata.obs["pred_matched_tissue"].values == tissue)

    ax.scatter(
        x[~pred_mask], y[~pred_mask],
        s=0.9, c=bg_color, edgecolors="none", linewidths=0, rasterized=True
    )
    ax.scatter(
        x[pred_mask], y[pred_mask],
        s=0.9, c=color, edgecolors="none", linewidths=0, rasterized=True
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

axes[0, 0].text(
    -0.26, 0.5, "ground truth",
    transform=axes[0, 0].transAxes,
    rotation=90, va="center", ha="center", fontsize=13
)
axes[1, 0].text(
    -0.26, 0.5, "Ours",
    transform=axes[1, 0].transAxes,
    rotation=90, va="center", ha="center", fontsize=13
)

plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:", out_png)
print("Saved:", out_pdf)
print("Saved:", out_csv)