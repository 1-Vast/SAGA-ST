#!/bin/bash
set -uo pipefail

echo "=================================================================="
echo "🚀 开始一键复现 DLPFC 151674 最佳结果 (目标 ARI: ~0.58, NMI: ~0.66)"
echo "=================================================================="

# ------------------------------------------------------------------
# 1. 强制设定严格的确定性环境 (非常重要，防随机种子抖动)
# ------------------------------------------------------------------
unset OMP_NUM_THREADS || true
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export NUMBA_THREADING_LAYER=workqueue
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------------
# 全局路径配置
# ------------------------------------------------------------------
H5_CLEAN="/root/autodl-tmp/DLPFC/151674_dropNaN.h5ad"
PREFIX="/root/autodl-tmp/DLPFC/151674_FINAL_BEST"
NPZ_FILE="${PREFIX}.augK2_d64_for_cluster.npz"

echo -e "\n>>> [1/3] 正在训练特征空间 (K=2, concat, k=15, gamma=2.0) ..."

# ------------------------------------------------------------------
# 2. 运行特征训练 (main.py)
# ------------------------------------------------------------------
python -m model.main \
  --h5 "$H5_CLEAN" \
  --out_prefix "$PREFIX" \
  --graph_model KNN --k 15 \
  --use_scanpy_workflow --pca_comps 64 \
  --dim 64 --K 2 --embed_agg concat --hidden 512 \
  --alpha 0.5 --topN 50 \
  --lambda_recon 0.05 --mask_ratio_feat 0.2 \
  --epochs 1200 --pos_per_epoch 20000 \
  --layer_aware --no_layer_fallback \
  --pseudo_layer_bins 7 --pseudo_layer_knn 20 \
  --neg_layer_margin 2 --layer_gamma 2.0 \
  --neg_hard_ratio 0.6 --neg_oversample 8 \
  --normal_aware --normal_knn 15 --normal_margin 1.0 --normal_gamma 2.0 \
  --activation prelu --scheduler \
  --lr 5e-4 --weight_decay 1e-4 \
  --seed 42 --device cuda

echo -e "\n>>> [2/3] 正在执行无损平滑聚类 (mclust, pca=30, smooth_k=6, refine) ..."

# ------------------------------------------------------------------
# 3. 运行聚类 (cluster.py)
# ------------------------------------------------------------------
python -m model.cluster \
  --npz "$NPZ_FILE" \
  --h5 "$H5_CLEAN" \
  --label_key sce.layer_guess \
  --method mclust \
  --use_rep emb \
  --pca_dim 30 \
  --n_clusters 7 \
  --smooth \
  --smooth_k 6 \
  --refine \
  --refine_iter 1 \
  --calc_acc \
  --progress

echo -e "\n>>> [3/3] 正在生成混淆矩阵 (Crosstab) 验证分层纯度 ..."

# ------------------------------------------------------------------
# 4. 自动计算并打印混淆矩阵
# ------------------------------------------------------------------
LABELS_FILE="$(dirname "$PREFIX")/cluster_results/$(basename "$PREFIX").augK2_d64_for_cluster.mclust.labels.txt"

python - <<PY
import anndata as ad
import pandas as pd
import os

h5_path = "$H5_CLEAN"
lab_path = "$LABELS_FILE"

if not os.path.exists(lab_path):
    print(f"找不到标签文件: {lab_path}")
    exit(1)

adata = ad.read_h5ad(h5_path)
pred = pd.read_csv(lab_path, sep="\t")

cid_col = "cell_id" if "cell_id" in pred.columns else pred.columns[0]
lbl_col = "cluster" if "cluster" in pred.columns else pred.columns[-1]

pred = pred[[cid_col, lbl_col]].copy()
pred.columns = ["cell_id", "pred"]
pred["cell_id"] = pred["cell_id"].astype(str)
pred["pred"] = pred["pred"].astype(str)

df = pd.DataFrame({
    "cell_id": adata.obs_names,
    "gt": adata.obs["sce.layer_guess"].astype(str)
})
df = df.merge(pred, on="cell_id", how="inner")
df = df[~df["gt"].isin(["na", "nan", "None"])]

crosstab = pd.crosstab(df["gt"], df["pred"])
print("\n=== Crosstab: GT x Pred ===")
print(crosstab)

row_norm = crosstab.div(crosstab.sum(axis=1), axis=0)
print("\n=== Row-normalized ===")
print(row_norm.round(3))
PY

echo "=================================================================="
echo "🎉 运行结束！最佳聚类可视化图保存在: $(dirname "$PREFIX")/cluster_results/"
echo "=================================================================="