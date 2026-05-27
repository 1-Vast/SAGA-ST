import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =========================
# 路径
# =========================
h5ad_path = "/root/autodl-tmp/Mouse/E14.5_E1S3.MOSTA.h5ad"
pred_label_path = "/root/autodl-tmp/Mouse/cluster_results/E14.5_E1S3_SSL_A_leiden.leiden.labels.txt"

out_path = "/root/autodl-tmp/Mouse/final_plot_color_no_green_brain.png"


# =========================
# 参数
# =========================
SPOT_SIZE_BG = 0.28
SPOT_SIZE_FG = 0.62
BG_COLOR = "#d3d5db"


# =========================
# cluster 映射
# =========================
PRED_MAP = {
    "Heart": ["13"],
    "Liver": ["11"],
    "Olfactory epithelium": ["14"],
    "Connective tissue": ["3"],
    "Muscle": ["9"],
    "Epidermis": ["2"],
    "Cavity": ["0"],          # 显示标题仍写 Meninges
    "Brain": ["1", "8"],
}

ORDER = [
    "Heart",
    "Liver",
    "Olfactory epithelium",
    "Connective tissue",
    "Muscle",
    "Epidermis",
    "Cavity",
    "Brain",
]

DISPLAY_TITLES = {
    "Heart": "Heart",
    "Liver": "Liver",
    "Olfactory epithelium": "Olfactory\nepithelium",
    "Connective tissue": "Connective\ntissue",
    "Muscle": "Muscle",
    "Epidermis": "Epidermis",
    "Cavity": "Meninges",
    "Brain": "Brain",
}

# 鲜明颜色
COLOR_MAP = {
    "Heart": "#f07f2f",
    "Liver": "#5666d6",
    "Olfactory epithelium": "#4a97e8",
    "Connective tissue": "#f2c230",
    "Muscle": "#e64980",
    "Epidermis": "#9b6acb",
    "Cavity": "#49d19e",
    "Brain": "#b48ae6",
}


# =========================
# 读 label
# =========================
def read_label_file(path, obs_names):
    obs_names = np.asarray(obs_names).astype(str)

    try:
        df = pd.read_csv(path, sep="\t")
        cols_lower = [str(c).lower() for c in df.columns]
        if "cluster" in cols_lower:
            cluster_col = df.columns[cols_lower.index("cluster")]
            labels = df[cluster_col].astype(str).values
            if len(labels) == len(obs_names):
                return pd.Series(labels, index=obs_names)
    except Exception:
        pass

    df = pd.read_csv(path, sep="\t", header=None)

    first_row = df.iloc[0].astype(str).str.lower().tolist()
    if ("cell_id" in first_row) or ("cluster" in first_row):
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] >= 2:
        labels = df.iloc[:, 1].astype(str).values
    else:
        labels = df.iloc[:, 0].astype(str).values

    if len(labels) != len(obs_names):
        raise ValueError(
            f"标签数 {len(labels)} 与 obs 数 {len(obs_names)} 不一致，请检查 labels 文件格式。"
        )

    return pd.Series(labels, index=obs_names)


# =========================
# GT 映射
# =========================
def normalize_gt(x):
    s = str(x).strip().lower()
    mapping = {
        "heart": "Heart",
        "liver": "Liver",
        "olfactory epithelium": "Olfactory epithelium",
        "cartilage primordium": "Connective tissue",
        "connective tissue": "Connective tissue",
        "muscle": "Muscle",
        "epidermis": "Epidermis",
        "meninges": "Cavity",
        "cavity": "Cavity",
        "brain": "Brain",
    }
    return mapping.get(s, str(x))


# =========================
# 构建 mask
# =========================
def build_gt_masks(gt):
    gt = gt.map(normalize_gt)
    masks = {}
    for name in ORDER:
        if name == "Brain":
            masks[name] = (gt == "Brain").values
        else:
            masks[name] = (gt == name).values
    return masks


def build_pred_masks(pred):
    masks = {}
    for name in ORDER:
        cls = PRED_MAP[name]
        masks[name] = pred.isin(cls).values
    return masks


# =========================
# 绘图函数
# =========================
def draw(ax, x, y, mask, title, color):
    ax.scatter(x, y, s=SPOT_SIZE_BG, c=BG_COLOR, linewidths=0, rasterized=True)

    if mask is not None and np.any(mask):
        ax.scatter(x[mask], y[mask], s=SPOT_SIZE_FG, c=color, linewidths=0, rasterized=True)

    ax.set_title(title, fontsize=15, pad=8)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


# =========================
# 主程序
# =========================
adata = ad.read_h5ad(h5ad_path)

pred = read_label_file(pred_label_path, adata.obs_names)
gt = adata.obs["annotation"].astype(str)

xy = adata.obsm["spatial"]
x, y = xy[:, 0], xy[:, 1]

gt_masks = build_gt_masks(gt)
pred_masks = build_pred_masks(pred)

fig = plt.figure(figsize=(16, 6), dpi=400)
gs = GridSpec(2, 8, figure=fig)

# 上排：8 个
top_axes = [fig.add_subplot(gs[0, i]) for i in range(8)]
for ax, name in zip(top_axes, ORDER):
    draw(ax, x, y, gt_masks[name], DISPLAY_TITLES[name], COLOR_MAP[name])

# 下排：8 个
bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(8)]
for ax, name in zip(bottom_axes, ORDER):
    draw(ax, x, y, pred_masks[name], DISPLAY_TITLES[name], COLOR_MAP[name])

# 左侧标签
fig.text(0.012, 0.72, "ground truth", rotation=90, va="center", ha="left", fontsize=16)
fig.text(0.012, 0.28, "Ours", rotation=90, va="center", ha="left", fontsize=16)

# 左上角 C
fig.text(0.02, 0.96, "C", fontsize=24, ha="left", va="top")

# 间距
plt.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.06, wspace=0.10, hspace=0.28)

plt.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:", out_path)