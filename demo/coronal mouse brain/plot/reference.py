import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

# =========================
# 1. 读取数据
# =========================
h5ad_path = r"D:\Spatial-main\dataset\coronal mouse brain\mouse_brain_with_allen_label.h5ad"
out_png = r"D:\test1\MuCoST_Fig3A_Reference_with_legend.png"

adata = sc.read_h5ad(h5ad_path)

label_key = "allen_cluster"

if label_key not in adata.obs.columns:
    raise KeyError(f"没有找到 {label_key}，当前 obs columns = {list(adata.obs.columns)}")

if "spatial" not in adata.obsm:
    raise KeyError(f"没有找到 adata.obsm['spatial']，当前 obsm keys = {list(adata.obsm.keys())}")

coords = adata.obsm["spatial"]
labels = adata.obs[label_key].astype("category")

# =========================
# 2. 指定图例顺序
# =========================
desired_order = [
    "Cortex_1",
    "Cortex_2",
    "Cortex_3",
    "Cortex_4",
    "Cortex_5",
    "Fiber_tract",
    "Hippocampus",
    "Hypothalamus_1",
    "Hypothalamus_2",
    "Lateral_ventricle",
    "Pyramidal_layer",
    "Pyramidal_layer_dentate_gyrus",
    "Striatum",
    "Thalamus_1",
    "Thalamus_2",
]

# 只保留数据中实际存在的类别
categories = [c for c in desired_order if c in set(labels)]

# 如果数据里还有额外类别，也加到最后
extra_categories = [c for c in labels.cat.categories if c not in categories]
categories = categories + extra_categories

# =========================
# 3. 尝试读取 H&E 背景图
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
        elif "lowres" in lib["images"]:
            img = lib["images"]["lowres"]

    if "scalefactors" in lib:
        if "tissue_hires_scalef" in lib["scalefactors"] and img is not None:
            scale = lib["scalefactors"]["tissue_hires_scalef"]
        elif "tissue_lowres_scalef" in lib["scalefactors"] and img is not None:
            scale = lib["scalefactors"]["tissue_lowres_scalef"]

except Exception as e:
    print("未读取到 H&E 背景图，将使用灰色背景。")
    print(e)

x = coords[:, 0] * scale
y = coords[:, 1] * scale

# =========================
# 4. 颜色设置
# =========================
# 尽量使用论文/Squidpy 常见颜色风格
manual_colors = {
    "Cortex_1": "#1f77b4",
    "Cortex_2": "#ff7f0e",
    "Cortex_3": "#2ca02c",
    "Cortex_4": "#d62728",
    "Cortex_5": "#a23bec",
    "Fiber_tract": "#8c564b",
    "Hippocampus": "#e377c2",
    "Hypothalamus_1": "#bcbd22",
    "Hypothalamus_2": "#17becf",
    "Lateral_ventricle": "#aec7e8",
    "Pyramidal_layer": "#ffbb78",
    "Pyramidal_layer_dentate_gyrus": "#98df8a",
    "Striatum": "#ff9896",
    "Thalamus_1": "#c5b0d5",
    "Thalamus_2": "#c49c94",
}

# 如果数据本身保存了颜色，也优先使用数据自带颜色
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
# 5. 绘图：左侧空间图 + 右侧图例
# =========================
fig = plt.figure(figsize=(6.0, 4.3), dpi=600)

# 左侧空间图区域
ax = fig.add_axes([0.06, 0.16, 0.52, 0.74])

if img is not None:
    ax.imshow(img)
else:
    ax.set_facecolor("#8f968d")

for cat in categories:
    mask = labels == cat
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

# 保留黑色边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 底部标题
fig.text(
    0.32,
    0.055,
    "Reference",
    fontsize=20,
    fontweight="bold",
    ha="center",
    va="center",
)

# =========================
# 6. 右侧图例
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

legend_ax = fig.add_axes([0.62, 0.12, 0.36, 0.80])
legend_ax.axis("off")

legend_ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
    fontsize=12,
    handletextpad=0.8,
    labelspacing=0.55,
    borderaxespad=0.0,
)

# =========================
# 7. 只保存 PNG
# =========================
plt.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print("已保存：", os.path.abspath(out_png))