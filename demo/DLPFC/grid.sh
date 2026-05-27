#!/usr/bin/env bash
set -euo pipefail

# ----- 路径配置（使用 tuned 嵌入） -----
NPZ="/root/autodl-tmp/DLPFC/151507_TUNED.augK2_d64_for_cluster.npz"
H5="/root/autodl-tmp/DLPFC/151507.h5ad"
LABEL="sce.layer_guess"

OUTROOT="/root/autodl-tmp/DLPFC/grid_tuned"
mkdir -p "$OUTROOT/cluster_results"

ROBUST_SMOOTH_ITER=1
FIXED_RES_LIST="0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72"

# ----- 搜索空间（围绕当前最佳值微调） -----
# 固定 knn_k 和 resolution（也可微调，但先聚焦主要参数）
KNNK=12
BASE_RES=0.58

# 空间权重微调范围
WSPA_LIST=(0.45 0.48 0.50 0.52 0.55)
# PCA 降维维度微调范围
PCA_DIM_LIST=(35 38 40 42 45)
# 空间 KNN 细化邻居数微调范围
REFINE_K_LIST=(30 33 35 38 40)
# 细化迭代次数（可固定为2或尝试3）
REFINE_ITER_LIST=(2 3)
# power 平滑保持关闭（因为之前最佳未用）
POWER_ALPHA=0.0

RESULTS_TSV="$OUTROOT/results.tsv"
echo -e "ARI\tNMI\tACC\tw_spa\tpca_dim\trefine_k\trefine_iter" > "$RESULTS_TSV"

run_one () {
  local wspa="$1"
  local pca_dim="$2"
  local rk="$3"
  local rit="$4"

  local wemb
  wemb=$(python -c "print(f'{1.0-float(\"$wspa\"):.2f}')")

  local tag="w${wspa}_pca${pca_dim}_rk${rk}_ri${rit}"
  local prefix="$OUTROOT/$tag"

  echo "[RUN] $tag"

  python -m model.cluster \
    --npz "$NPZ" \
    --h5  "$H5" \
    --label_key "$LABEL" \
    --method robust --n_clusters 7 \
    --knn_k "$KNNK" --resolution "$BASE_RES" \
    --robust_res_list "$FIXED_RES_LIST" \
    --w_spa "$wspa" --w_emb "$wemb" \
    --pca_dim "$pca_dim" --pca_key "X_pca_${pca_dim}" \
    --power 0 --power_alpha 0.0 \
    --robust_smooth_iter "$ROBUST_SMOOTH_ITER" \
    --refine_k "$rk" --refine_iter "$rit" \
    --calc_acc \
    --out_prefix "$prefix" \
    > "${prefix}.log" 2>&1

  # 解析 metrics 文件（注意路径）
  local mfile="${OUTROOT}/cluster_results/$(basename "$prefix").robust.metrics.txt"
  if [[ ! -f "$mfile" ]]; then
    echo "[WARN] metrics not found: $mfile"
    echo -e "NA\tNA\tNA\t$wspa\t$pca_dim\t$rk\t$rit" >> "$RESULTS_TSV"
    return 0
  fi

  local ari nmi acc
  ari=$(grep -i "ARI" "$mfile" | grep -oE "[0-9]+\.[0-9]+" | head -n 1 || echo "NA")
  nmi=$(grep -i "NMI" "$mfile" | grep -oE "[0-9]+\.[0-9]+" | head -n 1 || echo "NA")
  acc=$(grep -i "ACC" "$mfile" | grep -oE "[0-9]+\.[0-9]+" | head -n 1 || echo "NA")

  echo -e "${ari}\t${nmi}\t${acc}\t$wspa\t$pca_dim\t$rk\t$rit" >> "$RESULTS_TSV"
  echo "[DONE] ARI=${ari}  NMI=${nmi}  ACC=${acc}"
}

# 执行网格搜索
for wspa in "${WSPA_LIST[@]}"; do
  for pca_dim in "${PCA_DIM_LIST[@]}"; do
    for rk in "${REFINE_K_LIST[@]}"; do
      for rit in "${REFINE_ITER_LIST[@]}"; do
        run_one "$wspa" "$pca_dim" "$rk" "$rit"
      done
    done
  done
done

echo
echo "================ TOP 20 by ARI ================"
python - <<PY
import pandas as pd
df=pd.read_csv("$RESULTS_TSV", sep="\t")
df=df[df["ARI"].astype(str)!="NA"].copy()
df["ARI"]=df["ARI"].astype(float)
df=df.sort_values("ARI", ascending=False)
print(df.head(20).to_string(index=False))
PY