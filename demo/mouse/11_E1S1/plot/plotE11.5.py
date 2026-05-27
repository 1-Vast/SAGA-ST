import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# =============================
# 输入文件
# =============================
h5ad = "/root/autodl-tmp/Mouse/E11.5_E1S1.MOSTA.h5ad"
labels = "/root/autodl-tmp/demo/mouse/11_E1S1/E11.5_E1S1.label.txt"

# =============================
# 读取数据
# =============================
adata = sc.read_h5ad(h5ad)
lab = pd.read_csv(labels, sep="\t").set_index("cell_id")

adata = adata[adata.obs_names.intersection(lab.index)].copy()
adata.obs["cluster"] = lab.loc[adata.obs_names]["cluster"].astype(int)

coords = adata.obsm["spatial"]
x = coords[:,0]
y = coords[:,1]

# =============================
# 颜色（柔和论文配色）
# =============================
palette = [
"#d55e5e","#6b7f3a","#a78dd6","#4c9c6d","#2d6a4f",
"#5f5ab6","#7f6cc4","#e0893f","#9e2a2b","#d4a81e",
"#c95a7a","#c97dbd","#8d5a97","#4f81bd","#c7c27a",
"#9ccc65","#4db6c8","#9ecae1"
]

clusters = sorted(adata.obs.cluster.unique())
colors = {c: palette[i % len(palette)] for i,c in enumerate(clusters)}

# =============================
# 自动裁剪边界
# =============================
pad = 20
xmin,xmax = x.min()-pad, x.max()+pad
ymin,ymax = y.min()-pad, y.max()+pad

# =============================
# 绘图
# =============================
fig,ax = plt.subplots(figsize=(7,8),dpi=300)

for c in clusters:
    idx = adata.obs.cluster==c
    ax.scatter(
        x[idx],
        -y[idx],          # flip Y
        s=4.4,            
        color=colors[c],
        linewidths=0,
        alpha=0.9, 
        rasterized=True
    )

# 去掉坐标
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

ax.set_xlim(xmin,xmax)
ax.set_ylim(-ymax,-ymin)

ax.set_aspect("equal")

# =============================
# 图例
# =============================
handles = [
    Line2D([0],[0],
           marker='o',
           color='w',
           label=str(i+1),
           markerfacecolor=colors[c],
           markersize=18)
    for i,c in enumerate(clusters)
]

ax.legend(
    handles=handles,
    bbox_to_anchor=(1.02,0.5),
    loc="center left",
    frameon=False,
    ncol=2,
    columnspacing=1.2,
    handletextpad=0.3,
    fontsize=18
)

# =============================
# 标题
# =============================
ax.set_title("E11.5_E1S1 (ARI: 0.5080)",fontsize=16,pad=15)

plt.tight_layout()

plt.savefig("/root/autodl-tmp/E11.5_E1S1_paper_style.png",dpi=600,bbox_inches="tight")
plt.savefig("/root/autodl-tmp/E11.5_E1S1_paper_style.pdf",bbox_inches="tight")

print("Saved figure")