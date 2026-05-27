import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# =========================================================
# 1. 文件路径
# =========================================================
h5ad_path = "/root/autodl-tmp/Spatial-main/dataset/DLPFC/151672.h5ad"
label_path = "/root/autodl-tmp/Spatial-main/demo/DLPFC/151672/train/151672_v2.augK4_d96_for_cluster.mclust.labels.txt"

out_dir = Path("/root/autodl-tmp/paper_figures")
out_dir.mkdir(parents=True, exist_ok=True)

out_png = out_dir / "151672_layer_cluster_darkmatch.png"

# =========================================================
# 2. 读取数据
# =========================================================
adata = sc.read_h5ad(h5ad_path)
labels = pd.read_csv(label_path, sep="\t")

if "cell_id" not in labels.columns or "cluster" not in labels.columns:
    raise ValueError(
        f"标签文件列名不符合预期，当前列名为: {list(labels.columns)}；"
        f"需要包含 ['cell_id', 'cluster']"
    )

labels = labels.set_index("cell_id")

common = adata.obs_names.intersection(labels.index)
if len(common) == 0:
    raise ValueError("h5ad 与 labels.txt 没有共同的 cell_id / spot_id，请检查文件是否匹配。")

adata = adata[common].copy()
labels = labels.loc[common].copy()

adata.obs["cluster"] = labels["cluster"].astype(int)
adata.obs["cluster_show"] = adata.obs["cluster"] + 1

# =========================================================
# 3. 空间坐标
# =========================================================
if "spatial" not in adata.obsm:
    raise ValueError("adata.obsm 中没有 'spatial'，无法绘制空间图。")

adata.obs["x"] = adata.obsm["spatial"][:, 0]
adata.obs["y"] = adata.obsm["spatial"][:, 1]

# =========================================================
# 4. 颜色定义：对应 1~7
# =========================================================
cluster_order = [1, 2, 3, 4, 5, 6, 7]

# 使用统一风格的暗色调
cluster_colors = {
    1: "#4B78A8",  # 深蓝
    2: "#D48A45",  # 土橙
    3: "#6F9B57",  # 橄榄绿
    4: "#C85C5C",  # 砖红
    5: "#9A88C9",  # 灰紫
    6: "#8C6D5A",  # 棕褐
    7: "#B07AA1",  # 灰粉紫
}

# =========================================================
# 5. 方向调整
# =========================================================
flip_x = False
flip_y = False

adata.obs["x_plot"] = adata.obs["x"].copy()
adata.obs["y_plot"] = adata.obs["y"].copy()

if flip_x:
    adata.obs["x_plot"] = -adata.obs["x_plot"]

if flip_y:
    adata.obs["y_plot"] = -adata.obs["y_plot"]

# =========================================================
# 6. 绘图
# =========================================================
fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=300)
ax.set_facecolor("#CFCFCF")

for c in cluster_order:
    sub = adata.obs[adata.obs["cluster_show"] == c]
    if len(sub) == 0:
        continue

    ax.scatter(
        sub["x_plot"],
        sub["y_plot"],
        s=30,
        c=cluster_colors[c],
        edgecolors="#E6E6E6",
        linewidths=0.15
    )

# =========================================================
# 去掉坐标轴
# =========================================================
ax.set_xticks([])
ax.set_yticks([])

# 去掉边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 保持比例
ax.set_aspect("equal", adjustable="box")

# 标题
ax.set_title("151672", fontsize=16, pad=12)

# =========================================================
# 图例：显示 1~7
# =========================================================
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=str(c),
        markerfacecolor=cluster_colors[c],
        markeredgecolor="none",
        markersize=7
    )
    for c in cluster_order
]

ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=10
)

plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close()

print(f"图已保存到: {out_png}")