import os
from pathlib import Path
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. 路径
# =========================================================
h5ad_path = "/root/autodl-tmp/Brain (Coronal)/mouse_brain_with_allen_label.h5ad"
npz_path = "/root/autodl-tmp/Brain (Coronal)/GridSearch_SSL_30/SSL_Dim80_Alpha0.50_LR0.001.augK2_d80_for_cluster.npz"
label_path = "/root/autodl-tmp/Brain (Coronal)/GridSearch_SSL_30/cluster_results/SSL_Dim80_Alpha0.50_LR0.001.augK2_d80_for_cluster.mclust.labels.txt"

out_dir = Path("/root/autodl-tmp/Brain (Coronal)/paper_figures_coronal_v2")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. 参数
# =========================================================
group_key = "mclust15"
display_key = "cluster_show"
name_key = "cluster_name"

n_neighbors = 15
dotplot_top_n = 2
spot_size = 45          # 比上一版更小
umap_point_size = 12    # UMAP点更小
paga_threshold = 0.03

# =========================================================
# 3. 更暗一点的颜色
#    顺序对应 cluster 1~15
# =========================================================
palette_15_dark = [
    "#1F78B4",  # 1
    "#E68613",  # 2
    "#2CA02C",  # 3
    "#E31A1C",  # 4
    "#9146F0",  # 5
    "#8C564B",  # 6
    "#D96BB8",  # 7
    "#A9B34B",  # 8
    "#22B8CF",  # 9
    "#B7D0F6",  # 10
    "#E9B96E",  # 11
    "#8CD17D",  # 12
    "#F28E8E",  # 13
    "#B39DDB",  # 14
    "#C8B0AA",  # 15
]

# =========================================================
# 4. 工具函数
# =========================================================
def looks_like_counts(X, max_n=2000):
    if sp.issparse(X):
        Xs = X[:min(X.shape[0], max_n), :min(X.shape[1], max_n)].toarray()
    else:
        Xs = np.asarray(X[:min(X.shape[0], max_n), :min(X.shape[1], max_n)])
    if Xs.size == 0:
        return False
    if np.nanmin(Xs) < 0:
        return False
    frac = np.abs(Xs - np.round(Xs))
    return np.quantile(frac, 0.999) < 1e-3

def prepare_expr_for_markers(adata_in: ad.AnnData) -> ad.AnnData:
    adata = adata_in.copy()
    adata.var_names_make_unique()
    if looks_like_counts(adata.X):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata

def load_labels(label_file: str) -> pd.DataFrame:
    df = pd.read_csv(label_file, sep="\t")
    if "cell_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"标签文件列名异常: {df.columns.tolist()}")
    df["cell_id"] = df["cell_id"].astype(str)
    df["cluster"] = df["cluster"].astype(int)
    return df

def attach_embedding_from_npz(adata: ad.AnnData, npz_file: str):
    d = np.load(npz_file, allow_pickle=True)
    emb = d["embedding"]
    obs_names = d["obs_names"].astype(str)

    emb_df = pd.DataFrame(emb, index=obs_names)
    common = adata.obs_names.intersection(emb_df.index)
    adata = adata[common].copy()
    adata.obsm["emb"] = emb_df.loc[adata.obs_names].values.astype(np.float32)
    return adata

def attach_cluster_labels(adata: ad.AnnData, label_file: str, group_key: str):
    lab = load_labels(label_file).set_index("cell_id")
    common = adata.obs_names.intersection(lab.index)
    adata = adata[common].copy()

    adata.obs[group_key] = lab.loc[adata.obs_names, "cluster"].astype(int).astype(str).values
    adata.obs["cluster_id"] = lab.loc[adata.obs_names, "cluster"].astype(int).values
    adata.obs[display_key] = (adata.obs["cluster_id"] + 1).astype(int).astype(str)

    cat_order = [str(i) for i in range(1, 16)]
    adata.obs[display_key] = pd.Categorical(adata.obs[display_key], categories=cat_order, ordered=True)

    group_order = [str(i) for i in range(15)]
    adata.obs[group_key] = pd.Categorical(adata.obs[group_key], categories=group_order, ordered=True)
    return adata

def build_cluster_name_map(adata: ad.AnnData, group_key: str, ref_key: str = "allen_cluster"):
    """
    用 mclust cluster 对 allen_cluster 做多数投票，生成详细名字
    例如: 1 -> Cortex_1
    若重复，则自动加后缀 _2
    """
    df = adata.obs[[group_key, ref_key]].copy()
    mapping = {}
    used = defaultdict(int)

    for g in df[group_key].cat.categories:
        sub = df[df[group_key] == g]
        top_name = sub[ref_key].value_counts().index[0]
        used[top_name] += 1
        if used[top_name] == 1:
            mapping[g] = top_name
        else:
            mapping[g] = f"{top_name}_{used[top_name]}"
    return mapping

def get_top_markers(adata_marker: ad.AnnData, group_key: str, top_n: int = 2):
    sc.tl.rank_genes_groups(
        adata_marker,
        groupby=group_key,
        method="wilcoxon",
        n_genes=max(10, top_n),
        pts=True,
    )
    result = adata_marker.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    marker_dict = {}
    used = set()

    for g in groups:
        genes = [x for x in result["names"][g][:max(10, top_n)]]
        keep = []
        for gene in genes:
            if gene not in used:
                keep.append(gene)
                used.add(gene)
            if len(keep) >= top_n:
                break
        marker_dict[g] = keep

    return marker_dict

def plot_dotplot(adata_marker: ad.AnnData, marker_dict, cluster_name_map, out_png: str):
    rename_map = {}
    for k in marker_dict.keys():
        show_id = str(int(k) + 1)
        long_name = cluster_name_map[k]
        rename_map[k] = f"{show_id}"
    var_names = {rename_map[k]: v for k, v in marker_dict.items()}

    dp = sc.pl.dotplot(
        adata_marker,
        var_names=var_names,
        groupby=display_key,
        standard_scale="var",
        dendrogram=False,
        color_map="Reds",
        swap_axes=False,
        show=False,
        return_fig=True,
    )
    dp.savefig(out_png, dpi=300)
    plt.close("all")

def add_umap_text_labels(ax, adata_plot, color_key, name_key):
    um = adata_plot.obsm["X_umap"]
    df = pd.DataFrame({
        "x": um[:, 0],
        "y": um[:, 1],
        "cluster": adata_plot.obs[color_key].astype(str).values,
        "name": adata_plot.obs[name_key].astype(str).values,
    })

    for cl in sorted(df["cluster"].unique(), key=lambda x: int(x)):
        sub = df[df["cluster"] == cl]
        cx = sub["x"].median()
        cy = sub["y"].median()
        nm = sub["name"].iloc[0]
        ax.text(
            cx, cy, nm,
            fontsize=9.5,
            color="black",
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            zorder=10,
        )

def plot_umap(adata_plot: ad.AnnData, out_png: str):
    sc.pp.neighbors(adata_plot, use_rep="emb", n_neighbors=n_neighbors)
    sc.tl.umap(adata_plot, random_state=0)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    sc.pl.umap(
        adata_plot,
        color=display_key,
        palette=palette_15_dark,
        size=umap_point_size,
        frameon=False,
        legend_loc=None,
        title="UMAP (mclust)",
        ax=ax,
        show=False,
    )

    add_umap_text_labels(ax, adata_plot, display_key, name_key)

    handles = [
        Line2D([0], [0], marker="o", color="w", label=str(i + 1),
               markerfacecolor=palette_15_dark[i], markeredgecolor="none", markersize=8)
        for i in range(15)
    ]
    ax.legend(
        handles=handles,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_paga(adata_plot: ad.AnnData, cluster_name_map, out_png: str):
    if "neighbors" not in adata_plot.uns:
        sc.pp.neighbors(adata_plot, use_rep="emb", n_neighbors=n_neighbors)

    sc.tl.paga(adata_plot, groups=display_key)

    # 自定义节点标签
    paga_labels = {str(i + 1): cluster_name_map[str(i)] for i in range(15)}

    fig = plt.figure(figsize=(10.5, 6.5))
    sc.pl.paga(
        adata_plot,
        color=display_key,
        labels=paga_labels,
        threshold=paga_threshold,
        node_size_scale=2.6,
        edge_width_scale=1.15,
        frameon=False,
        fontsize=9.5,
        show=False,
    )
    plt.title("PAGA trajectory (mclust)", fontsize=16, pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_spatial_pub(adata_plot: ad.AnnData, out_png: str):
    xy = np.asarray(adata_plot.obsm["spatial"])
    x = xy[:, 0]
    y = xy[:, 1]
    labels = adata_plot.obs[display_key].astype(str).values

    color_map = {str(i + 1): palette_15_dark[i] for i in range(15)}
    c = [color_map[v] for v in labels]

    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.set_facecolor("#8B8F88")  # 更稳重一点的灰背景

    ax.scatter(
        x, y,
        c=c,
        s=spot_size,
        edgecolors="none",
        linewidths=0,
        alpha=0.97,
    )

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=str(i + 1),
            markerfacecolor=palette_15_dark[i],
            markeredgecolor="none",
            markersize=6.8,
        )
        for i in range(15)
    ]

    ax.legend(
        handles=handles,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        handletextpad=0.45,
        columnspacing=1.0,
        fontsize=10,
    )

    ax.set_title("Spatial domains (mclust)", fontsize=18, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 5. 主流程
# =========================================================
print("Loading h5ad...")
adata = sc.read_h5ad(h5ad_path)
adata.var_names_make_unique()

print("Attaching embedding from npz...")
adata = attach_embedding_from_npz(adata, npz_path)

print("Attaching cluster labels...")
adata = attach_cluster_labels(adata, label_path, group_key)

cluster_name_map = build_cluster_name_map(adata, group_key=group_key, ref_key="allen_cluster")
adata.obs[name_key] = adata.obs[group_key].astype(str).map(cluster_name_map).values

print("Cluster detailed names:")
for k in sorted(cluster_name_map.keys(), key=lambda x: int(x)):
    print(f"{int(k)+1}: {cluster_name_map[k]}")

adata_plot = adata.copy()

print("Preparing expression for marker ranking...")
adata_marker = prepare_expr_for_markers(adata)
adata_marker.obs[group_key] = adata.obs[group_key].copy()
adata_marker.obs[display_key] = adata.obs[display_key].copy()
adata_marker.obs[name_key] = adata.obs[name_key].copy()

print("Ranking marker genes...")
marker_dict = get_top_markers(adata_marker, group_key=group_key, top_n=dotplot_top_n)
print("Top markers per cluster:")
for k, v in marker_dict.items():
    print(f"cluster {int(k)+1}: {v}")

# =========================================================
# 6. 出图
# =========================================================
dotplot_png = out_dir / "coronal_mclust_marker_dotplot_v2.png"
umap_png = out_dir / "coronal_mclust_umap_v2.png"
paga_png = out_dir / "coronal_mclust_paga_v2.png"
spatial_png = out_dir / "coronal_mclust_spatial_pub_v2.png"

print("Plotting dotplot...")
plot_dotplot(adata_marker, marker_dict, cluster_name_map, str(dotplot_png))

print("Plotting UMAP...")
plot_umap(adata_plot, str(umap_png))

print("Plotting PAGA...")
plot_paga(adata_plot, cluster_name_map, str(paga_png))

print("Plotting spatial publication-style figure...")
plot_spatial_pub(adata_plot, str(spatial_png))

print("\nSaved files:")
print(dotplot_png)
print(umap_png)
print(paga_png)
print(spatial_png)