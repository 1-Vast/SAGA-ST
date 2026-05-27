
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.csgraph import minimum_spanning_tree

BASE = Path('/root/autodl-tmp/coronal mouse brain/EffectPlots_3_final')
BASE.mkdir(parents=True, exist_ok=True)
H5 = Path('/root/autodl-tmp/coronal mouse brain/mouse_brain_with_allen_label.h5ad')
NPZ = Path('/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/d80_a055_lr001_k12_top36_l016_m010_e500.augK2_d80_for_cluster.npz')
LABELS = Path('/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/final_ari0658/cluster_results/d80_a055_p14_sklearn_fullcov_merge20.mclust.labels.txt')

adata = sc.read_h5ad(H5)
adata.var_names_make_unique()
z = np.load(NPZ, allow_pickle=True)
obs_names = z['obs_names'].astype(str)
if not np.array_equal(obs_names, adata.obs_names.astype(str)):
    adata = adata[obs_names].copy()
adata.obsm['emb'] = z['embedding'].astype(np.float32)
labels_df = pd.read_csv(LABELS, sep='\t')
labels = labels_df['cluster'].astype(str).to_numpy() if 'cluster' in labels_df.columns else labels_df.iloc[:, -1].astype(str).to_numpy()
adata.obs['Cluster'] = pd.Categorical(labels)
adata.obs['Allen_label'] = adata.obs['allen_cluster'].astype('category')

adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.neighbors(adata, use_rep='emb', n_neighbors=15)
sc.tl.umap(adata, random_state=0)
sc.tl.diffmap(adata)
spatial = np.asarray(adata.obsm['spatial'])
root = int(np.argmin(spatial[:, 0]))
adata.uns['iroot'] = root
sc.tl.dpt(adata, n_dcs=10)

plt.rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 320,
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#303030',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 01: Dotplot with only 6 labels and smaller points.
sc.tl.rank_genes_groups(adata, groupby='Cluster', method='wilcoxon', pts=True, use_raw=True)
markers = sc.get.rank_genes_groups_df(adata, group=None)
markers.to_csv(BASE / 'effect_marker_table.csv', index=False)
# Pick six largest final labels for a compact, readable plot.
cluster_order = adata.obs['Cluster'].value_counts().sort_values(ascending=False).head(6).index.astype(str).tolist()
selected = []
for cl in cluster_order:
    sub = markers[markers['group'].astype(str) == cl].sort_values(['pvals_adj', 'scores'], ascending=[True, False])
    for gene in sub['names'].head(3):
        if gene not in selected:
            selected.append(gene)
selected = selected[:18]
X = adata.raw[:, selected].X
if sp.issparse(X):
    X = X.toarray()
expr = pd.DataFrame(X, index=adata.obs_names, columns=selected)
expr['Cluster'] = adata.obs['Cluster'].astype(str).values
mean = expr.groupby('Cluster')[selected].mean().reindex(cluster_order)
frac = expr.groupby('Cluster')[selected].apply(lambda x: (x > 0).mean() * 100).reindex(cluster_order)
scaled = pd.DataFrame(MinMaxScaler(feature_range=(0.12, 1.0)).fit_transform(mean), index=mean.index, columns=mean.columns)
fig, ax = plt.subplots(figsize=(8.4, 3.35))
y_labels = [f'Label_{i+1}' for i in range(len(cluster_order))]
for yi, cl in enumerate(cluster_order):
    for xi, gene in enumerate(selected):
        size = 4 + float(frac.loc[cl, gene]) * 0.82
        color_val = float(scaled.loc[cl, gene])
        ax.scatter(xi, yi, s=size, c=[[1.0, 0.93 - 0.55*color_val, 0.88 - 0.80*color_val]], edgecolors='#bdaaa2', linewidths=0.25)
ax.set_xlim(-0.8, len(selected)-0.2)
ax.set_ylim(len(cluster_order)-0.45, -0.55)
ax.set_xticks(range(len(selected)))
ax.set_xticklabels(selected, rotation=90, ha='center', fontsize=8)
ax.set_yticks(range(len(cluster_order)))
ax.set_yticklabels(y_labels, fontsize=9)
ax.tick_params(length=2.5, color='#303030')
for s in ax.spines.values():
    s.set_visible(True)
    s.set_linewidth(0.8)
legend_sizes = [20, 40, 60, 80, 100]
handles = [plt.scatter([], [], s=4+s*0.82, color='#777777', edgecolors='none') for s in legend_sizes]
leg1 = ax.legend(handles, [str(s) for s in legend_sizes], title='Fraction of cells\nin group (%)', frameon=False,
                 loc='upper left', bbox_to_anchor=(1.02, 0.98), borderaxespad=0, labelspacing=0.85, handletextpad=0.7)
ax.add_artist(leg1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cax = fig.add_axes([0.865, 0.24, 0.095, 0.052])
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_ticks([0.35, 0.85])
cb.set_ticklabels(['2', '4'])
cax.set_title('Mean expression\nin group', fontsize=8, pad=7)
fig.subplots_adjust(left=0.08, right=0.79, bottom=0.33, top=0.94)
fig.savefig(BASE / '01_marker_dotplot.png', bbox_inches='tight')
plt.close(fig)

# Shared palette.
allen = adata.obs['Allen_label'].cat
cats = list(allen.categories)
palette = list(plt.cm.tab20.colors)
colors = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
labels_allen = adata.obs['Allen_label'].astype(str).to_numpy()

# 02: UMAP visualization, not spatial coordinates.
umap = np.asarray(adata.obsm['X_umap'])
fig, ax = plt.subplots(figsize=(6.2, 5.0))
for c in cats:
    idx = labels_allen == c
    ax.scatter(umap[idx, 0], umap[idx, 1], s=7.5, color=colors[c], linewidths=0, alpha=0.9)
# annotate main groups with small offsets; no panel letter.
major = adata.obs['Allen_label'].value_counts().head(11).index.tolist()
offsets = {
    'Hypothalamus_1': (0.0, 0.35), 'Thalamus_2': (-0.45, 0.25), 'Striatum': (0.35, 0.22),
    'Cortex_2': (0.38, 0.18), 'Cortex_4': (0.42, 0.05), 'Cortex_1': (0.35, -0.14),
    'Cortex_3': (0.22, -0.28), 'Fiber_tract': (0.0, -0.32), 'Hippocampus': (0.0, -0.38),
    'Thalamus_1': (-0.45, -0.05), 'Lateral_ventricle': (0.0, 0.18)
}
for c in major:
    idx = labels_allen == c
    if idx.sum() == 0: continue
    x, y = np.median(umap[idx], axis=0)
    dx, dy = offsets.get(c, (0, 0))
    txt = ax.text(x+dx, y+dy, c, ha='center', va='center', fontsize=8.2, weight='bold', color='#171717')
    txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground='white')])
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect('equal', adjustable='datalim')
fig.savefig(BASE / '02_umap_domains.png', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# 03: Trajectory graph with no right colorbar.
plot_xy = np.asarray(adata.obsm['X_umap'])
cent = {c: np.median(plot_xy[labels_allen == c], axis=0) for c in cats}
conn = adata.obsp['connectivities'].tocsr()
mat = pd.DataFrame(0.0, index=cats, columns=cats)
rows, cols = conn.nonzero()
for i, j in zip(rows, cols):
    if i >= j: continue
    a, b = labels_allen[i], labels_allen[j]
    if a == b: continue
    w = float(conn[i, j])
    mat.loc[a, b] += w
    mat.loc[b, a] += w
sizes = adata.obs['Allen_label'].value_counts().reindex(cats).astype(float)
for a in cats:
    for b in cats:
        if a != b:
            mat.loc[a, b] = mat.loc[a, b] / np.sqrt(sizes[a] * sizes[b])
coords = np.array([cent[c] for c in cats])
dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
mst = minimum_spanning_tree(dists).toarray()
edges = set()
for i in range(len(cats)):
    for j in range(len(cats)):
        if mst[i, j] > 0 or mst[j, i] > 0:
            edges.add(tuple(sorted((cats[i], cats[j]))))
strong_pairs = []
for i, a in enumerate(cats):
    for j, b in enumerate(cats):
        if i < j and mat.loc[a, b] > 0:
            strong_pairs.append((mat.loc[a, b], a, b))
for _, a, b in sorted(strong_pairs, reverse=True)[:6]:
    edges.add(tuple(sorted((a, b))))
ptime = adata.obs.groupby('Allen_label', observed=True)['dpt_pseudotime'].median().reindex(cats).to_numpy()
pos = {c: cent[c].astype(float) for c in cats}
manual = {
    'Lateral_ventricle': np.array([-0.40, 0.10]), 'Fiber_tract': np.array([0.0, -0.22]),
    'Striatum': np.array([0.30, 0.28]), 'Cortex_2': np.array([0.42, 0.32]),
    'Cortex_5': np.array([0.48, 0.42]), 'Cortex_4': np.array([0.48, 0.06]),
    'Cortex_1': np.array([0.38, -0.18]), 'Cortex_3': np.array([0.18, -0.34]),
    'Hippocampus': np.array([0.0, -0.45]), 'Pyramidal_layer': np.array([0.12, -0.55]),
    'Pyramidal_layer_dentate_gyrus': np.array([-0.18, -0.58]), 'Hypothalamus_1': np.array([0.0, 0.48]),
    'Thalamus_2': np.array([-0.42, 0.34]), 'Thalamus_1': np.array([-0.45, -0.08]),
}
for c, off in manual.items():
    if c in pos:
        pos[c] = pos[c] + off
edge_vals = np.array([mat.loc[a, b] if mat.loc[a, b] > 0 else 0.001 for a, b in edges])
fig, ax = plt.subplots(figsize=(6.2, 5.0))
for a, b in sorted(edges):
    w = mat.loc[a, b] if mat.loc[a, b] > 0 else edge_vals.min()
    lw = 0.75 + 2.8 * (w - edge_vals.min()) / (edge_vals.max() - edge_vals.min() + 1e-9)
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color='#252525', lw=lw, alpha=0.72, zorder=1)
node_xy = np.array([pos[c] for c in cats])
# normalize pseudotime color but no colorbar.
ax.scatter(node_xy[:, 0], node_xy[:, 1], c=ptime, cmap='viridis', s=145, edgecolor='white', linewidth=1.2, zorder=3)
for c in cats:
    txt = ax.text(pos[c][0], pos[c][1], c, ha='center', va='center', fontsize=8.0, weight='bold', color='#111111', zorder=4)
    txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground='white')])
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect('equal', adjustable='datalim')
fig.savefig(BASE / '03_trajectory_graph.png', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

(Path(BASE) / 'make_three_effect_plots.py').write_text(Path(__file__).read_text())
print('DONE', BASE)
