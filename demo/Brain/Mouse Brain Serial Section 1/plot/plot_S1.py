import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import to_hex, hsv_to_rgb

# ================================
# 1. 输入输出路径
# ================================
h5ad_path = "/root/autodl-tmp/Spatial-main/demo/Brain/Mouse Brain Serial Section 1/train/S1_ANT_POS.target_style.final.h5ad"
label_path = "/root/autodl-tmp/brain/cluster_results/S1_V4_mclust26_clean.mclust.labels.txt"

out_png = "/root/autodl-tmp/brain/S1/S1_ANT_POS_MERGED_highlight.png"
out_pdf = "/root/autodl-tmp/brain/S1/S1_ANT_POS_MERGED_highlight.pdf"

# ================================
# 2. 读取数据
# ================================
adata = sc.read_h5ad(h5ad_path)
labels = pd.read_csv(label_path, sep="\t")

if "cell_id" in labels.columns:
    labels = labels.set_index("cell_id")
else:
    raise ValueError("标签文件缺少 cell_id 列")

label_col = None
for c in labels.columns:
    if c.lower() in {"label", "labels", "cluster"}:
        label_col = c
        break
if label_col is None:
    raise ValueError("标签文件中未找到 label/labels/cluster 列")

common = adata.obs_names.intersection(labels.index)
if len(common) == 0:
    raise ValueError("h5ad 与标签文件没有重叠的 spot/cell id")

adata = adata[common].copy()
labels = labels.loc[common].copy()

adata.obs["cluster"] = labels[label_col].astype(int).astype(str).values

# ================================
# 3. 坐标处理
# ================================
xy = np.asarray(adata.obsm["spatial"]).copy()

center = xy.mean(axis=0, keepdims=True)
xy = xy - center

mirror_x = False
mirror_y = False

if mirror_x:
    xy[:, 0] = -xy[:, 0]
if mirror_y:
    xy[:, 1] = -xy[:, 1]

xy[:, 0] -= xy[:, 0].min()
xy[:, 1] -= xy[:, 1].min()

adata.obsm["spatial_plot"] = xy

# ================================
# 4. 高饱和鲜明配色
# ================================
cluster_ids = sorted(adata.obs["cluster"].unique(), key=lambda x: int(x))
n_clusters = len(cluster_ids)

strong_colors = [
    "#1f77b4",  # 0 强蓝
    "#d62728",  # 1 红
    "#ff7f0e",  # 2 橙
    "#2ca02c",  # 3 绿
    "#9467bd",  # 4 紫
    "#8c564b",  # 5 棕
    "#e41a1c",  # 6 亮红
    "#377eb8",  # 7 亮蓝
    "#4daf4a",  # 8 亮绿
    "#984ea3",  # 9 亮紫
    "#ff8c00",  # 10 深橙
    "#a65628",  # 11 深棕
    "#f781bf",  # 12 洋红
    "#7f7f7f",  # 13 深灰
    "#17becf",  # 14 青
    "#bcbd22",  # 15 黄绿
    "#393b79",  # 16 深蓝紫
    "#637939",  # 17 橄榄绿
    "#8c6d31",  # 18 金棕
    "#843c39",  # 19 砖红
    "#7b4173",  # 20 紫红
    "#3182bd",  # 21 强蓝
    "#31a354",  # 22 强绿
    "#e6550d",  # 23 强橙红
    "#756bb1",  # 24 靛紫
    "#636363",  # 25 深灰
    "#b30000",  # 26 暗红
    "#006d2c",  # 27 暗绿
    "#08519c",  # 28 暗蓝
    "#a50f15",  # 29 酒红
]

palette = strong_colors[:]

if n_clusters > len(palette):
    extra_n = n_clusters - len(palette)
    extra_colors = []
    hues = np.linspace(0, 1, extra_n, endpoint=False)
    for h in hues:
        rgb = hsv_to_rgb((h, 0.85, 0.85))
        extra_colors.append(to_hex(rgb))
    palette.extend(extra_colors)

palette = palette[:n_clusters]

# ================================
# 5. 手动高亮指定 cluster
#    把需要突出的紫色区域换成特别醒目的颜色
#    这里默认先把 cluster "22" 改成亮洋红
#    如果实际不是 22，改成对应编号即可
# ================================
highlight_map = {
    "22": "#ff2ca8",   # 亮洋红，最醒目
    # 例如还想改别的可以继续加：
    # "8": "#f4b400",
    # "20": "#00c853",
}

for i, cid in enumerate(cluster_ids):
    if cid in highlight_map:
        palette[i] = highlight_map[cid]

adata.uns["cluster_colors"] = palette

# 打印 cluster 和颜色对应关系，方便你确认
print("Cluster to color mapping:")
for cid, col in zip(cluster_ids, palette):
    print(f"{cid}\t{col}")

# ================================
# 6. 绘图
# ================================
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)

sc.pl.embedding(
    adata,
    basis="spatial_plot",
    color="cluster",
    palette=palette,
    size=70,
    frameon=True,
    legend_loc="right margin",
    legend_fontsize=9,
    legend_fontoutline=0,
    ax=ax,
    show=False,
    title="Section 1"
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=400, bbox_inches="tight")
plt.savefig(out_pdf, dpi=400, bbox_inches="tight")
plt.close()

print(f"Saved PNG: {out_png}")
print(f"Saved PDF: {out_pdf}")