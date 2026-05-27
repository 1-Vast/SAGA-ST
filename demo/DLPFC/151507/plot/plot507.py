import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# =========================================================
# 1. 文件路径
# =========================================================
h5ad_path = "/root/autodl-tmp/Spatial-main/dataset/DLPFC/151507.h5ad"
label_path = "/root/autodl-tmp/DLPFC/151507_tuning_v10/cluster_results/run_k10_a0.55_lr0.10_mr0.08.augK2_d64_for_cluster.robust.labels.txt"

out_dir = Path("/root/autodl-tmp/paper_figures")
out_dir.mkdir(parents=True, exist_ok=True)

out_png = out_dir / "151507_layer_cluster.png"

# =========================================================
# 2. 读取数据
# =========================================================
adata = sc.read_h5ad(h5ad_path)
labels = pd.read_csv(label_path, sep="\t")

labels = labels.set_index("cell_id")

common = adata.obs_names.intersection(labels.index)

adata = adata[common].copy()
labels = labels.loc[common].copy()

adata.obs["cluster"] = labels["cluster"].astype(int)

# =========================================================
# 3. 空间坐标
# =========================================================
adata.obs["x"] = adata.obsm["spatial"][:, 0]
adata.obs["y"] = adata.obsm["spatial"][:, 1]

# =========================================================
# 4. cluster 转为 1..7
# =========================================================
adata.obs["cluster_id"] = adata.obs["cluster"] + 1

cluster_order = sorted(adata.obs["cluster_id"].unique())

# =========================================================
# 5. 颜色
# =========================================================
cluster_colors = {
    1: "#4E79A7",
    2: "#F28E2B",
    3: "#59A14F",
    4: "#E15759",
    5: "#9C83C3",
    6: "#9C755F",
    7: "#D98BC3",
}

# =========================================================
# 6. 方向
# =========================================================
flip_x = False
flip_y = False

adata.obs["x_plot"] = adata.obs["x"]
adata.obs["y_plot"] = adata.obs["y"]

if flip_x:
    adata.obs["x_plot"] = -adata.obs["x_plot"]

if flip_y:
    adata.obs["y_plot"] = -adata.obs["y_plot"]

# =========================================================
# 7. 绘图
# =========================================================
fig, ax = plt.subplots(figsize=(7.2,7.2), dpi=300)

ax.set_facecolor("#d9d9d9")

for c in cluster_order:

    sub = adata.obs[adata.obs["cluster_id"] == c]

    ax.scatter(
        sub["x_plot"],
        sub["y_plot"],
        s=34,
        c=cluster_colors[c],
        edgecolors="none"
    )

# =========================================================
# 去坐标轴
# =========================================================
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_aspect("equal")

ax.set_title("151507", fontsize=16)

# =========================================================
# 图例 1..7
# =========================================================
legend_handles = [

    Line2D(
        [0],[0],
        marker="o",
        color="w",
        label=str(c),
        markerfacecolor=cluster_colors[c],
        markersize=7
    )

    for c in cluster_order

]

ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.02,0.5),
    frameon=False
)

plt.tight_layout()

# =========================================================
# 保存
# =========================================================
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close()

print("保存位置:")
print(out_png)