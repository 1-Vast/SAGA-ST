
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
from matplotlib.patches import Circle

BASE = Path('/root/autodl-tmp/coronal mouse brain/ClusterPlot_Adjusted')
BASE.mkdir(parents=True, exist_ok=True)
H5 = Path('/root/autodl-tmp/coronal mouse brain/mouse_brain_with_allen_label.h5ad')
LABELS = Path('/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/final_ari0658/cluster_results/d80_a055_p14_sklearn_fullcov.mclust.labels.txt')

adata = sc.read_h5ad(H5)
labels_df = pd.read_csv(LABELS, sep='\t')
if 'cluster' in labels_df.columns:
    labels = labels_df['cluster'].astype(int).to_numpy()
else:
    labels = labels_df.iloc[:, -1].astype(int).to_numpy()

spatial = np.asarray(adata.obsm['spatial'], dtype=float)
x = spatial[:, 0]
y = -spatial[:, 1]
# Use 1-based labels for legend, matching the reference figure.
plot_labels = labels + 1

palette = {
    1: '#1f77b4',  2: '#ff7f0e',  3: '#2ca02c',  4: '#d62728',  5: '#a23bec',
    6: '#8c564b',  7: '#e377c2',  8: '#a6b84f',  9: '#17becf', 10: '#aec7e8',
    11: '#ffbb78', 12: '#98df8a', 13: '#ff9896', 14: '#c5b0d5', 15: '#c49c94',
}
colors = [palette[int(v)] for v in plot_labels]

fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.set_facecolor('#8d938b')
fig.patch.set_facecolor('white')

# faint array/background dots, like the reference histology/grid backdrop
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
px = np.linspace(xmin - 0.03*(xmax-xmin), xmax + 0.03*(xmax-xmin), 42)
py = np.linspace(ymin - 0.03*(ymax-ymin), ymax + 0.03*(ymax-ymin), 42)
gx, gy = np.meshgrid(px, py)
ax.scatter(gx.ravel(), gy.ravel(), s=7, facecolors='none', edgecolors='#626862', linewidths=0.35, alpha=0.55, zorder=0)

ax.scatter(x, y, c=colors, s=13, edgecolors='#565b58', linewidths=0.25, alpha=0.96, zorder=2)

# White arrows with numbered gray circles, placed by relative figure coordinates.
def data_at(rx, ry):
    return xmin + rx * (xmax - xmin), ymin + ry * (ymax - ymin)

arrow_specs = [
    # number, circle xy, arrow start, arrow end
    (1, (0.39, 0.82), (0.47, 0.82), (0.47, 0.70)),
    (2, (0.43, 0.52), (0.49, 0.50), (0.49, 0.62)),
    (3, (0.72, 0.57), (0.71, 0.57), (0.61, 0.57)),
    (4, (0.66, 0.34), (0.66, 0.32), (0.66, 0.45)),
    (5, (0.50, 0.14), (0.50, 0.16), (0.50, 0.25)),
    (6, (0.17, 0.30), (0.24, 0.32), (0.24, 0.18)),
]
for num, circle_r, start_r, end_r in arrow_specs:
    cx, cy = data_at(*circle_r)
    sx, sy = data_at(*start_r)
    ex, ey = data_at(*end_r)
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy), arrowprops=dict(arrowstyle='-|>', color='white', lw=2.6, mutation_scale=15), zorder=4)
    radius = 0.038 * (xmax - xmin)
    circ = Circle((cx, cy), radius=radius, facecolor=(0.55, 0.55, 0.55, 0.55), edgecolor='white', linewidth=1.0, zorder=5)
    ax.add_patch(circ)
    ax.text(cx, cy, str(num), ha='center', va='center', color='white', fontsize=9, zorder=6)

ax.set_aspect('equal')
pad_x = 0.035 * (xmax - xmin)
pad_y = 0.035 * (ymax - ymin)
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#303030')
    spine.set_linewidth(1.0)

handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=palette[i], markeredgecolor='none', markersize=6, label=str(i)) for i in range(1, 16)]
leg = ax.legend(handles=handles, ncol=2, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), columnspacing=1.6, handletextpad=0.8, labelspacing=1.05, borderaxespad=0.0, fontsize=10)

fig.subplots_adjust(left=0.03, right=0.75, bottom=0.03, top=0.98)
fig.savefig(BASE / '04_cluster_spatial_adjusted.png', bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

(Path(BASE) / 'make_cluster_spatial_adjusted.py').write_text(Path(__file__).read_text())
print('DONE', BASE / '04_cluster_spatial_adjusted.png')
