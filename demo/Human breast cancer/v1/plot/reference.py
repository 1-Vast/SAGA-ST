import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# =========================
# 1. 路径设置
# =========================
h5ad_path = r"D:\Spatial-main\dataset\Human_Breast_Cancer\breast_with_gt.h5ad"
out_png = r"D:\test1\Breast_Reference_with_legend.png"

adata = sc.read_h5ad(h5ad_path)

print(adata)
print("obs columns:", list(adata.obs.columns))
print("obsm keys:", list(adata.obsm.keys()))
print("uns keys:", list(adata.uns.keys()))

# =========================
# 2. 自动寻找 Reference 标签列
# =========================
candidate_label_keys = [
    "fine_annot_type",
    "annotation",
    "ground_truth",
    "groundtruth",
    "label",
    "cluster",
    "domain",
]

label_key = None
for key in candidate_label_keys:
    if key in adata.obs.columns:
        label_key = key
        break

if label_key is None:
    raise KeyError(
        "没有找到标签列。请从打印出的 obs columns 中选择真实标签列，"
        "然后手动设置 label_key。"
    )

print("Using label_key =", label_key)
print(adata.obs[label_key].value_counts())

if "spatial" not in adata.obsm:
    raise KeyError(f"没有找到 adata.obsm['spatial']，当前 obsm keys = {list(adata.obsm.keys())}")

coords = adata.obsm["spatial"]
labels = adata.obs[label_key].astype("category")

# =========================
# 3. 图例顺序
# =========================
desired_order = [
    "DCIS/LCIS_1",
    "DCIS/LCIS_2",
    "DCIS/LCIS_4",
    "DCIS/LCIS_5",
    "Healthy_1",
    "Healthy_2",
    "IDC_1",
    "IDC_2",
    "IDC_3",
    "IDC_4",
    "IDC_5",
    "IDC_6",
    "IDC_7",
    "IDC_8",
    "Tumor_edge_1",
    "Tumor_edge_2",
    "Tumor_edge_3",
    "Tumor_edge_4",
    "Tumor_edge_5",
    "Tumor_edge_6",
]

existing = set(labels.astype(str))
categories = [c for c in desired_order if c in existing]

# 如果数据中还有其他标签，追加到最后
extra_categories = [c for c in labels.cat.categories if c not in categories]
categories = categories + list(extra_categories)

# =========================
# 4. 尝试读取组织背景图
# =========================
img = None
scale = 1.0

try:
    spatial_dict = adata.uns["spatial"]
    library_id = list(spatial_dict.keys())[0]
    lib = spatial_dict[library_id]

    if "images" in lib:
        if "hires" in lib["images"]:
            img = lib["images"]["hires"]
            if "scalefactors" in lib and "tissue_hires_scalef" in lib["scalefactors"]:
                scale = lib["scalefactors"]["tissue_hires_scalef"]
        elif "lowres" in lib["images"]:
            img = lib["images"]["lowres"]
            if "scalefactors" in lib and "tissue_lowres_scalef" in lib["scalefactors"]:
                scale = lib["scalefactors"]["tissue_lowres_scalef"]

except Exception as e:
    print("未读取到 H&E 背景图，将使用灰色背景。")
    print(e)

x = coords[:, 0] * scale
y = coords[:, 1] * scale

# =========================
# 5. 颜色设置
# =========================
manual_colors = {
    "DCIS/LCIS_1": "#1f77b4",
    "DCIS/LCIS_2": "#ff7f0e",
    "DCIS/LCIS_4": "#2ca02c",
    "DCIS/LCIS_5": "#d62728",
    "Healthy_1": "#a23bec",
    "Healthy_2": "#8c564b",
    "IDC_1": "#e377c2",
    "IDC_2": "#bcbd22",
    "IDC_3": "#17becf",
    "IDC_4": "#aec7e8",
    "IDC_5": "#ffbb78",
    "IDC_6": "#98df8a",
    "IDC_7": "#ff9896",
    "IDC_8": "#c5b0d5",
    "Tumor_edge_1": "#c49c94",
    "Tumor_edge_2": "#f7b6d2",
    "Tumor_edge_3": "#dbdb8d",
    "Tumor_edge_4": "#9edae5",
    "Tumor_edge_5": "#ad494a",
    "Tumor_edge_6": "#8c6d31",
}

color_key = f"{label_key}_colors"

if color_key in adata.uns:
    stored_categories = list(labels.cat.categories)
    stored_colors = list(adata.uns[color_key])
    color_dict = dict(zip(stored_categories, stored_colors))

    for c in categories:
        if c not in color_dict:
            color_dict[c] = manual_colors.get(c, "#333333")
else:
    color_dict = {
        c: manual_colors.get(c, plt.get_cmap("tab20")(i % 20))
        for i, c in enumerate(categories)
    }

# =========================
# 6. 绘图：左侧 Reference 图 + 右侧图例
# =========================
fig = plt.figure(figsize=(7.2, 4.7), dpi=600)

# 左侧空间图区域
ax = fig.add_axes([0.06, 0.16, 0.55, 0.76])

if img is not None:
    ax.imshow(img)
else:
    ax.set_facecolor("#b0b0b0")

for cat in categories:
    mask = labels.astype(str) == cat
    ax.scatter(
        x[mask],
        y[mask],
        s=5,
        c=[color_dict[cat]],
        linewidths=0,
        alpha=0.95,
    )

# 裁剪到组织区域
pad = 80
ax.set_xlim(x.min() - pad, x.max() + pad)
ax.set_ylim(y.max() + pad, y.min() - pad)

# 去掉坐标轴
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

# 保留边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 底部 Reference
fig.text(
    0.335,
    0.055,
    "Reference",
    fontsize=20,
    fontweight="bold",
    ha="center",
    va="center",
)

# =========================
# 7. 右侧图例
# =========================
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor=color_dict[cat],
        markeredgecolor="none",
        label=cat,
    )
    for cat in categories
]

legend_ax = fig.add_axes([0.64, 0.14, 0.34, 0.76])
legend_ax.axis("off")

legend_ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
    fontsize=11,
    handletextpad=0.8,
    labelspacing=0.45,
    borderaxespad=0.0,
    ncol=2,
    columnspacing=1.4,
)

# =========================
# 8. 只保存 PNG
# =========================
plt.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print("已保存：", os.path.abspath(out_png))