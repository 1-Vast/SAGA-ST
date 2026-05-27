import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# =========================================================
# 1. 文件路径
# =========================================================
h5ad_path = "/root/autodl-tmp/Spatial-main/dataset/DLPFC/151508.h5ad"
label_path = "/root/autodl-tmp/Spatial-main/demo/DLPFC/151508/train/151508_TUNED_try1.robust.labels.txt"

# 输出目录
out_dir = Path("/root/autodl-tmp/paper_figures")
out_dir.mkdir(parents=True, exist_ok=True)

# 输出文件
out_png = out_dir / "151508_robust_layer_style_darkmatch.png"

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

# 对齐 spot
common = adata.obs_names.intersection(labels.index)
if len(common) == 0:
    raise ValueError("h5ad 与 labels.txt 没有共同的 cell_id / spot_id，请检查文件是否匹配。")

adata = adata[common].copy()
labels = labels.loc[common].copy()

adata.obs["cluster"] = labels["cluster"].astype(int).astype(str)

# =========================================================
# 3. 空间坐标
# =========================================================
if "spatial" not in adata.obsm:
    raise ValueError("adata.obsm 中没有 'spatial'，无法绘制空间图。")

adata.obs["x"] = adata.obsm["spatial"][:, 0]
adata.obs["y"] = adata.obsm["spatial"][:, 1]

# =========================================================
# 4. cluster -> layer 映射
# ---------------------------------------------------------
# 先按默认顺序设置
# 若画出来层顺序和人工标注不一致，只改这里即可
# =========================================================
cluster_to_layer = {
    "0": "Layer1",
    "1": "Layer2",
    "2": "Layer3",
    "3": "Layer4",
    "4": "Layer5",
    "5": "Layer6",
    "6": "WM"
}

adata.obs["layer_name"] = adata.obs["cluster"].map(cluster_to_layer)

unmapped = adata.obs["layer_name"].isna().sum()
if unmapped > 0:
    missing_clusters = sorted(adata.obs.loc[adata.obs["layer_name"].isna(), "cluster"].unique())
    raise ValueError(f"以下 cluster 没有映射到层名: {missing_clusters}")

# =========================================================
# 5. 颜色定义
# =========================================================
layer_order = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]

layer_colors = {
    "Layer1": "#4B78A8",  # 深蓝
    "Layer2": "#D48A45",  # 土橙
    "Layer3": "#6F9B57",  # 橄榄绿
    "Layer4": "#C85C5C",  # 砖红
    "Layer5": "#9A88C9",  # 灰紫
    "Layer6": "#8C6D5A",  # 棕褐
    "WM":     "#B07AA1"   # 灰粉紫
}

# =========================================================
# 6. 方向调整
# ---------------------------------------------------------
# 按你的要求：flip_y = False
# 如果后面发现左右方向不对，再把 flip_x 改成 True
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

# 背景色
ax.set_facecolor("#CFCFCF")

for layer in layer_order:
    sub = adata.obs[adata.obs["layer_name"] == layer]
    if len(sub) == 0:
        continue

    ax.scatter(
        sub["x_plot"],
        sub["y_plot"],
        s=30,
        c=layer_colors[layer],
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
ax.set_title("151508", fontsize=16, pad=12)

# =========================================================
# 图例
# =========================================================
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=layer,
        markerfacecolor=layer_colors[layer],
        markeredgecolor="none",
        markersize=7
    )
    for layer in layer_order
]

ax.legend(
    handles=legend_handles,
    title="Layer",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()

# =========================================================
# 保存
# =========================================================
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close()

print(f"图已保存到: {out_png}")