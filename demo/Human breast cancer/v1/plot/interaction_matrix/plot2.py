import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# =========================================================
# 1. 路径与参数
# =========================================================
h5_path = "/root/autodl-tmp/cancer/breast_with_gt.h5ad"
label_path = "/root/autodl-tmp/cancer/cluster_results/breast_v1.augK3_d96_for_cluster.robust.labels.txt"

out_dir = "/root/autodl-tmp/cancer/interaction_matrix"
os.makedirs(out_dir, exist_ok=True)

prefix = os.path.join(out_dir, "breast_robust_target_style")

cluster_col = "robust_cluster"
knn_k = 6
fig_dpi = 500

# 目标图风格参数
target_max = 30.0          # colorbar 上限做成 0~30
clip_quantile = 0.97       # 裁剪高值，减弱超强对角线
figsize = (5.8, 5.0)
cmap = "viridis"

# =========================================================
# 2. 工具函数
# =========================================================
def build_spatial_knn_graph(spatial: np.ndarray, k: int = 6) -> sp.csr_matrix:
    n = spatial.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
    nn.fit(spatial)
    indices = nn.kneighbors(return_distance=False)

    rows, cols = [], []
    for i in range(n):
        neigh = indices[i, 1:]  # 去掉自己
        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    )
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def compute_interaction_counts(A: sp.csr_matrix, labels: np.ndarray, uniq_labels: np.ndarray) -> np.ndarray:
    label_to_idx = {lab: i for i, lab in enumerate(uniq_labels)}
    K = len(uniq_labels)
    M = np.zeros((K, K), dtype=np.float64)

    coo = A.tocoo()
    for i, j in zip(coo.row, coo.col):
        if i < j:  # 无向边只统计一次
            a = label_to_idx[labels[i]]
            b = label_to_idx[labels[j]]
            M[a, b] += 1
            M[b, a] += 1
    return M


def transform_to_target_style(M_counts: np.ndarray, target_max: float = 30.0, clip_quantile: float = 0.97):
    """
    让图更接近目标论文风格：
    1) log1p 压缩动态范围
    2) 按分位数裁剪高值
    3) 线性缩放到 0~target_max
    """
    X = np.log1p(M_counts.astype(np.float64))

    positive = X[X > 0]
    if positive.size == 0:
        return X

    clip_val = np.quantile(positive, clip_quantile)
    if clip_val <= 0:
        clip_val = positive.max()

    X = np.clip(X, 0, clip_val)
    X = X / clip_val * target_max
    return X


def save_matrix_csv(M: np.ndarray, uniq_labels: np.ndarray, out_csv: str):
    df = pd.DataFrame(M, index=uniq_labels, columns=uniq_labels)
    df.to_csv(out_csv)
    print(f"saved: {out_csv}")


def plot_target_style_heatmap(
    M: np.ndarray,
    uniq_labels: np.ndarray,
    out_png: str,
    vmin: float = 0.0,
    vmax: float = 30.0,
    cmap: str = "viridis",
    figsize=(5.8, 5.0),
    dpi: int = 500,
):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(len(uniq_labels)))
    ax.set_yticks(range(len(uniq_labels)))
    ax.set_xticklabels(uniq_labels, fontsize=9)
    ax.set_yticklabels(uniq_labels, fontsize=9)

    ax.tick_params(axis="both", length=3, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_png}")


# =========================================================
# 3. 读数据并合并标签
# =========================================================
print("Loading h5ad...")
adata = ad.read_h5ad(h5_path)

print("Loading labels...")
lab = pd.read_csv(label_path, sep="\t")
lab["cell_id"] = lab["cell_id"].astype(str)

adata.obs[cluster_col] = -1
common = adata.obs_names.intersection(lab["cell_id"])
mapper = lab.set_index("cell_id")["cluster"].astype(int)
adata.obs.loc[common, cluster_col] = mapper.loc[common].values

labels = adata.obs[cluster_col].to_numpy().astype(int)
if np.any(labels < 0):
    raise ValueError("存在 spot 没有成功匹配 cluster 标签，请检查 obs_names 与 labels 文件。")

uniq_labels = np.sort(np.unique(labels))
print("Unique clusters:", uniq_labels.tolist())

# =========================================================
# 4. 构建空间图
# =========================================================
print("Building spatial KNN graph...")
spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
A = build_spatial_knn_graph(spatial, k=knn_k)
print("A shape:", A.shape, "nnz:", A.nnz)

# =========================================================
# 5. 计算 interaction counts
# =========================================================
print("Computing interaction counts...")
M_counts = compute_interaction_counts(A, labels, uniq_labels)

# 保存原始 counts
save_matrix_csv(M_counts, uniq_labels, prefix + ".counts.csv")

# 做成接近目标论文图的数值矩阵
M_target = transform_to_target_style(
    M_counts,
    target_max=target_max,
    clip_quantile=clip_quantile,
)

save_matrix_csv(M_target, uniq_labels, prefix + ".target_scaled.csv")

# =========================================================
# 6. 画图
# =========================================================
plot_target_style_heatmap(
    M_target,
    uniq_labels,
    prefix + ".png",
    vmin=0.0,
    vmax=target_max,
    cmap=cmap,
    figsize=figsize,
    dpi=fig_dpi,
)

print("\nDone.")
print("Output image:", prefix + ".png")
print("Counts csv   :", prefix + ".counts.csv")
print("Scaled csv   :", prefix + ".target_scaled.csv")