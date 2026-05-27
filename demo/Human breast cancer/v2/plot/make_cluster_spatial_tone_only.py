
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path('/root/autodl-tmp/coronal mouse brain/ClusterPlot_ToneOnly')
BASE.mkdir(parents=True, exist_ok=True)
H5 = Path('/root/autodl-tmp/coronal mouse brain/mouse_brain_with_allen_label.h5ad')
LABELS = Path('/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/final_ari0658/cluster_results/d80_a055_p14_sklearn_fullcov.mclust.labels.txt')

adata = sc.read_h5ad(H5)
labels_df = pd.read_csv(LABELS, sep='\t')
labels = labels_df['cluster'].astype(int).to_numpy() if 'cluster' in labels_df.columns else labels_df.iloc[:, -1].astype(int).to_numpy()
xy = np.asarray(adata.obsm['spatial'], dtype=float)
x = xy[:, 0]
y = -xy[:, 1]
plot_labels = labels + 1

# Soft but distinct manuscript-style palette close to the reference color family.
palette = {
    1: '#2b83ba',  2: '#f28e2b',  3: '#59a14f',  4: '#d33f49',  5: '#a349d6',
    6: '#8c5a4a',  7: '#d65db1',  8: '#a7ad3f',  9: '#21b6c7', 10: '#9ebcda',
    11: '#fdb462', 12: '#8bd17c', 13: '#fb8072', 14: '#b39ddb', 15: '#c49a8c',
}
colors = [palette[int(v)] for v in plot_labels]

plt.rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 340,
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.0,
})

fig, ax = plt.subplots(figsize=(6.0, 5.2))
fig.patch.set_facecolor('white')
ax.set_facecolor('#8f968d')

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
# Subtle full-array background, only for formatting texture.
px = np.linspace(xmin - 0.025*(xmax-xmin), xmax + 0.025*(xmax-xmin), 45)
py = np.linspace(ymin - 0.025*(ymax-ymin), ymax + 0.025*(ymax-ymin), 45)
gx, gy = np.meshgrid(px, py)
ax.scatter(gx.ravel(), gy.ravel(), s=6, facecolors='none', edgecolors='#677067', linewidths=0.32, alpha=0.42, zorder=0)

# Larger, cleaner points with subtle dark outline.
ax.scatter(x, y, c=colors, s=12.5, edgecolors='#484f4a', linewidths=0.22, alpha=0.97, zorder=2)

ax.set_aspect('equal')
pad_x = 0.035 * (xmax - xmin)
pad_y = 0.035 * (ymax - ymin)
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#2f332f')
    spine.set_linewidth(1.05)

handles = [Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=palette[i], markeredgecolor='none', markersize=6.5, label=str(i)) for i in range(1, 16)]
ax.legend(
    handles=handles,
    ncol=2,
    frameon=False,
    loc='center left',
    bbox_to_anchor=(1.035, 0.5),
    columnspacing=1.7,
    handletextpad=0.75,
    labelspacing=1.08,
    borderaxespad=0,
    fontsize=10,
)
fig.subplots_adjust(left=0.035, right=0.75, bottom=0.035, top=0.985)
fig.savefig(BASE / '05_cluster_spatial_tone_only.png', bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

(Path(BASE) / 'make_cluster_spatial_tone_only.py').write_text(Path(__file__).read_text())
print('DONE', BASE / '05_cluster_spatial_tone_only.png')
