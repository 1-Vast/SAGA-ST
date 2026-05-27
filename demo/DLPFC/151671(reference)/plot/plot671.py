import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# =========================================================
# 1. 文件路径
# =========================================================
h5ad_path = "/root/autodl-tmp/Spatial-main/dataset/DLPFC/151671.h5ad"
label_path = "/root/autodl-tmp/DLPFC/search_50_trials/cluster_results/trial_21.augK2_d64_for_cluster.robust.labels.txt"

out_dir = Path("/root/autodl-tmp/paper_figures")
out_dir.mkdir(parents=True, exist_ok=True)

out_png = out_dir / "151671_layer_cluster_darkmatch.png"

# =========================================================
# 2. 读取数据
# =========================================================
adata = sc.read_h5ad(h5ad_path)
labels = pd.read_csv(label_path, sep="\t")

if "cell_id" not in labels.columns or "cluster" not in labels.columns:
    raise ValueError(
        f"标签文件列名不符合预期，当前列名为: {list(labels.columns)}；需要包含 ['cell_id', 'cluster']"
    )

labels = labels.set_index("cell_id")
common = adata.obs_names.intersection(labels.index)

if len(common) == 0:
    raise ValueError("h5ad 与 labels.txt 没有共同的 cell_id / spot_id")

adata = adata[common].copy()
labels = labels.loc[common].copy()

adata.obs["cluster"] = labels["cluster"].astype(int)

# =========================================================
# 3. 把实际出现的 cluster 重映射为连续的 1,2,3,...
# =========================================================
unique_clusters = sorted(adata.obs["cluster"].unique())
cluster_remap = {old: new for new, old in enumerate(unique_clusters, start=1)}
adata.obs["cluster_show"] = adata.obs["cluster"].map(cluster_remap)

cluster_order = list(range(1, len(unique_clusters) + 1))

# =========================================================
# 4. 空间坐标
# =========================================================
if "spatial" not in adata.obsm:
    raise ValueError("adata.obsm 中没有 spatial 坐标")

adata.obs["x"] = adata.obsm["spatial"][:, 0]
adata.obs["y"] = adata.obsm["spatial"][:, 1]

# =========================================================
# 5. 更暗、更贴近参考图的颜色
# =========================================================
# 对应视觉顺序：
# 1 -> 蓝（Layer3）
# 2 -> 橙（Layer4）
# 3 -> 绿（Layer5）
# 4 -> 红（Layer6）
# 5 -> 紫（WM）
base_colors = [
    "#4B78A8",  # 深一点的蓝
    "#D48A45",  # 土橙
    "#6F9B57",  # 橄榄绿
    "#C85C5C",  # 砖红
    "#9A88C9",  # 灰紫
    "#8C6D5A",  # 备用
    "#B07AA1",  # 备用
]

if len(cluster_order) > len(base_colors):
    raise ValueError("类别数超过预设颜色数，请补充颜色。")

cluster_colors = {c: base_colors[c - 1] for c in cluster_order}

# =========================================================
# 6. 方向调整
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
# 7. 绘图
# =========================================================
fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=300)

# 背景再暗一点，更接近参考图
ax.set_facecolor("#CFCFCF")

for c in cluster_order:
    sub = adata.obs[adata.obs["cluster_show"] == c]
    if len(sub) == 0:
        continue

    ax.scatter(
        sub["x_plot"],
        sub["y_plot"],
        s=34,
        c=cluster_colors[c],
        edgecolors="none"
    )

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_aspect("equal")
ax.set_title("151671", fontsize=16, pad=12)

# =========================================================
# 8. 图例
# =========================================================
legend_handles = [
    Line2D(
        [0], [0],
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
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=10
)

plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close()

print("图已保存到:")
print(out_png)
print("原始 cluster -> 显示编号:", cluster_remap)