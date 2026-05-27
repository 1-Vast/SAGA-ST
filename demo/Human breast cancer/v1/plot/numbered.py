import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


h5ad_path = r"D:\Spatial-main\dataset\Human_Breast_Cancer\breast_with_gt.h5ad"
npz_path = r"D:\Spatial-main\demo\Human breast cancer\train\breast_v1.augK3_d96_for_cluster.npz"
label_path = r"D:\Spatial-main\demo\Human breast cancer\train\breast_v1.augK3_d96_for_cluster.robust.labels.txt"
out_png = r"D:\test1\Breast_MuCoST_domain_numbered.png"

DISPLAY_PLUS_ONE = True

os.makedirs(os.path.dirname(out_png), exist_ok=True)

adata = sc.read_h5ad(h5ad_path)

if "spatial" not in adata.obsm:
    raise KeyError(f"没有找到 adata.obsm['spatial']，当前 obsm keys = {list(adata.obsm.keys())}")

npz = np.load(npz_path, allow_pickle=True)

if "obs_names" in npz:
    npz_obs_names = npz["obs_names"].astype(str)
else:
    npz_obs_names = np.array(adata.obs_names).astype(str)


def read_cluster_labels(path):
    df = pd.read_csv(
        path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        comment="#"
    )

    df = df.dropna(how="all")
    numeric_df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    numeric_counts = numeric_df.notna().sum(axis=0)

    label_col = numeric_counts.idxmax()
    valid_mask = numeric_df[label_col].notna()

    labels = numeric_df.loc[valid_mask, label_col].astype(int).to_numpy()

    if df.shape[1] >= 2 and label_col != 0:
        names = df.loc[valid_mask, 0].astype(str).to_numpy()
    else:
        names = None

    return names, labels


label_names, cluster_labels = read_cluster_labels(label_path)

if label_names is not None and len(label_names) == len(cluster_labels):
    label_series = pd.Series(cluster_labels, index=label_names.astype(str))
    common = [x for x in adata.obs_names.astype(str) if x in label_series.index]

    if len(common) == 0:
        raise ValueError("labels.txt 中的 spot id 与 adata.obs_names 没有交集。")

    adata_plot = adata[common].copy()
    raw_labels = label_series.loc[common].to_numpy()

else:
    if len(cluster_labels) != len(npz_obs_names):
        raise ValueError(
            f"标签数量 {len(cluster_labels)} 与 npz_obs_names 数量 {len(npz_obs_names)} 不一致。"
        )

    label_series = pd.Series(cluster_labels, index=npz_obs_names.astype(str))
    common = [x for x in adata.obs_names.astype(str) if x in label_series.index]

    if len(common) == 0:
        raise ValueError("adata.obs_names 与 npz_obs_names 没有交集。")

    adata_plot = adata[common].copy()
    raw_labels = label_series.loc[common].to_numpy()

coords = adata_plot.obsm["spatial"]

if DISPLAY_PLUS_ONE and raw_labels.min() == 0:
    show_labels = raw_labels + 1
else:
    show_labels = raw_labels.copy()

print("聚类标签分布：")
print(pd.Series(raw_labels).value_counts().sort_index())

print("图中显示编号：")
print(sorted(np.unique(show_labels)))

img = None
scale = 1.0

try:
    spatial_dict = adata_plot.uns["spatial"]
    library_id = list(spatial_dict.keys())[0]
    lib = spatial_dict[library_id]

    if "images" in lib:
        if "hires" in lib["images"]:
            img = lib["images"]["hires"]
            scale = lib.get("scalefactors", {}).get("tissue_hires_scalef", 1.0)
        elif "lowres" in lib["images"]:
            img = lib["images"]["lowres"]
            scale = lib.get("scalefactors", {}).get("tissue_lowres_scalef", 1.0)

except Exception as e:
    print("未读取到背景图，使用灰色背景。")
    print(e)

x = coords[:, 0] * scale
y = coords[:, 1] * scale

base_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
]

unique_show = sorted(np.unique(show_labels))

number_to_color = {
    num: base_colors[i % len(base_colors)]
    for i, num in enumerate(unique_show)
}

fig = plt.figure(figsize=(4.8, 4.8), dpi=600)
ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])

if img is not None:
    ax.imshow(img)
else:
    ax.set_facecolor("#b0b0b0")

for num in unique_show:
    mask = show_labels == num
    ax.scatter(
        x[mask],
        y[mask],
        s=5,
        c=[number_to_color[num]],
        linewidths=0,
        alpha=0.90,
    )

pad = 80
ax.set_xlim(x.min() - pad, x.max() + pad)
ax.set_ylim(y.max() + pad, y.min() - pad)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

for num in unique_show:
    mask = show_labels == num

    cx = np.median(x[mask])
    cy = np.median(y[mask])

    dx = x[mask] - cx
    dy = y[mask] - cy
    idx_local = np.argmin(dx * dx + dy * dy)

    cx = x[mask][idx_local]
    cy = y[mask][idx_local]

    ax.text(
        cx,
        cy,
        str(num),
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="black",
        path_effects=[
            pe.withStroke(linewidth=2.0, foreground="white")
        ],
    )

plt.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print("已保存：", os.path.abspath(out_png))