import os
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANT = "/root/autodl-tmp/Brain/Mouse Brain Serial Section 2/Anterior/V1_Mouse_Brain_Sagittal_Anterior_Section_2.h5ad"
POS = "/root/autodl-tmp/Brain/Mouse Brain Serial Section 2/Posterior/V1_Mouse_Brain_Sagittal_Posterior_Section_2.h5ad"

OUT_DIR = "/root/autodl-tmp/Brain/S2_ANT_POS_MERGED"
OUT_H5  = f"{OUT_DIR}/S2_ANT_POS.target_style.final.h5ad"
OUT_PNG = f"{OUT_DIR}/preview.target_style.final.png"

GAP_X = 300  # 和S1一致（需要更大缝就调这个）

def get_xy(adata):
    # 优先用全分辨率像素坐标（更贴近 Visium 的“真实图像坐标”）
    if ("pxl_col_in_fullres" in adata.obs) and ("pxl_row_in_fullres" in adata.obs):
        x = adata.obs["pxl_col_in_fullres"].to_numpy().astype(np.float32)
        y = adata.obs["pxl_row_in_fullres"].to_numpy().astype(np.float32)
        return np.stack([x, y], axis=1)
    return np.asarray(adata.obsm["spatial"], dtype=np.float32)

def rot90_cw(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.stack([y, -x], axis=1)

def flip_ud(xy):
    xy2 = xy.copy()
    xy2[:, 1] = -xy2[:, 1]
    return xy2

def centerize(xy):
    return xy - xy.mean(axis=0, keepdims=True)

print("[1] read ...")
a = sc.read_h5ad(ANT)
p = sc.read_h5ad(POS)

# ---- 基础清理：和S1一致 ----
a.var_names_make_unique()
p.var_names_make_unique()

a.obs_names = ["ANT_" + str(x) for x in a.obs_names]
p.obs_names = ["POS_" + str(x) for x in p.obs_names]
a.obs_names_make_unique()
p.obs_names_make_unique()

a.obs["batch"] = "ANT"
p.obs["batch"] = "POS"

# ---- 坐标：同S1的 target-style 变换链 ----
xy_a = centerize(get_xy(a))
xy_p = centerize(get_xy(p))

xy_a = flip_ud(rot90_cw(xy_a))
xy_p = flip_ud(rot90_cw(xy_p))

# 平移到正坐标域（避免负数影响可视化/后续）
xy_a[:, 0] -= xy_a[:, 0].min()
xy_a[:, 1] -= xy_a[:, 1].min()
xy_p[:, 0] -= xy_p[:, 0].min()
xy_p[:, 1] -= xy_p[:, 1].min()

# 右移 POS，形成拼接
xy_p[:, 0] += xy_a[:, 0].max() + GAP_X

# ---- 取共同基因交集并 concat ----
common = np.intersect1d(a.var_names, p.var_names)
a = a[:, common].copy()
p = p[:, common].copy()

# 去掉各自的 spatial uns（避免 Visium 的 nested dict 冲突）
if "spatial" in a.uns: del a.uns["spatial"]
if "spatial" in p.uns: del p.uns["spatial"]

m = ad.concat([a, p], join="inner")
m.obsm["spatial"] = np.vstack([xy_a, xy_p])

# ======= 关键：整体上下翻转（和S1 final一致）=======
m.obsm["spatial"][:, 1] *= -1

os.makedirs(OUT_DIR, exist_ok=True)
m.write(OUT_H5)
print("[OK] saved:", OUT_H5)

# ---- 预览图 ----
plt.figure(figsize=(14, 5))
xy = m.obsm["spatial"]
b = m.obs["batch"].values
for lab in ["ANT", "POS"]:
    mask = (b == lab)
    plt.scatter(xy[mask, 0], xy[mask, 1], s=6, label=lab)
plt.legend()
plt.axis("equal")
plt.title("S2 Target-style merge preview (final flipped)")
plt.savefig(OUT_PNG, dpi=240)
print("[OK] saved preview:", OUT_PNG)
