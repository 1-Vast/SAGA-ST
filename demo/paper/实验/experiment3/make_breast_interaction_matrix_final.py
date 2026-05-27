
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
counts = pd.read_csv(base / 'breast_robust_target_style.counts.csv', index_col=0).astype(float)
M = (counts.to_numpy(float) + counts.to_numpy(float).T) / 2.0
np.fill_diagonal(M, 0.0)
row = M.sum(axis=1, keepdims=True)
col = M.sum(axis=0, keepdims=True)
expected = row @ col / max(M.sum(), 1.0)
oe = np.divide(M, expected, out=np.zeros_like(M), where=expected > 0)
S = np.log2(oe + 1.0)
np.fill_diagonal(S, np.nan)
vals = S[np.isfinite(S) & (S > 0)]
vmax = max(float(np.percentile(vals, 96)) if vals.size else 1.0, 1.0)
cmap = plt.cm.viridis.copy()
cmap.set_bad('#eeeeee')
plt.rcParams.update({'figure.dpi': 180, 'savefig.dpi': 420, 'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 1.0})
fig, ax = plt.subplots(figsize=(7.0, 6.2))
im = ax.imshow(S, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest', aspect='equal')
n = S.shape[0]
labels = [str(i + 1) for i in range(n)]
ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
ax.tick_params(axis='both', length=2.5, width=0.8, pad=2.5)
ax.set_xlabel('Spatial domain', fontsize=9, labelpad=6)
ax.set_ylabel('Spatial domain', fontsize=9, labelpad=6)
ax.set_title('Inter-domain adjacency enrichment', fontsize=9.5, pad=6)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.05); sp.set_color('black')
cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.032)
cbar.set_label('log2(O/E + 1)', fontsize=8.5, labelpad=6)
cbar.ax.tick_params(labelsize=7, length=2.5, width=0.8)
cbar.outline.set_linewidth(0.9)
fig.subplots_adjust(left=0.105, right=0.875, bottom=0.105, top=0.92)
fig.savefig(base / 'breast_robust_target_style.png', bbox_inches='tight', pad_inches=0.035)
pd.DataFrame(S, index=labels, columns=labels).to_csv(base / 'breast_robust_target_style.log2_oe_matrix.csv')
