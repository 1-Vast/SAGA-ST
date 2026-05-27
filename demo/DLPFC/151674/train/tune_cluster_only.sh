#!/bin/bash
set -uo pipefail

# 之前跑出的最高分特征文件
NPZ_FILE="/root/autodl-tmp/DLPFC/151674_BASELINE.augK2_d64_for_cluster.npz"
H5_FILE="/root/autodl-tmp/DLPFC/151674_dropNaN.h5ad"

# 输出目录
OUT_DIR="/root/autodl-tmp/DLPFC/cluster_finetune"
mkdir -p "$OUT_DIR/cluster_results"

LOG_FILE="$OUT_DIR/cluster_scan_summary.tsv"
echo -e "PCA_DIM\tTARGET_K\tMERGE\tFINAL_K\tARI\tNMI\tACC" > "$LOG_FILE"

# 扫描空间：极低维度到适中维度，以及超聚类 (8-9类)
PCA_LIST=(18 20 22 24 26 28 30)
N_CLUST_LIST=(7 8 9)
# 选项 1: 不合并  选项 2: 合并小于 30 的微小类 (释放被噪声占据的坑位)
MERGE_FLAGS=("" "--merge_small --min_cluster_size 30")

total_runs=$(( ${#PCA_LIST[@]} * ${#N_CLUST_LIST[@]} * ${#MERGE_FLAGS[@]} ))
current=1

echo "=================================================================="
echo "开始对最优特征进行聚类参数极限扫描 (总计 $total_runs 个组合)..."
echo "=================================================================="

for pca in "${PCA_LIST[@]}"; do
  for k in "${N_CLUST_LIST[@]}"; do
    for merge_opt in "${MERGE_FLAGS[@]}"; do
      
      merge_tag="NoMerge"
      if [[ "$merge_opt" != "" ]]; then
        merge_tag="Merge30"
      fi

      PREFIX="$OUT_DIR/pca${pca}_k${k}_${merge_tag}"
      echo ">>> [$current/$total_runs] 测试: PCA=${pca}, N_CLUST=${k}, MERGE=${merge_tag}"

      # 运行聚类代码
      python -m model.cluster \
        --npz "$NPZ_FILE" \
        --h5 "$H5_FILE" \
        --label_key "sce.layer_guess" \
        --method mclust \
        --use_rep emb \
        --pca_dim "$pca" \
        --n_clusters "$k" \
        --smooth --smooth_k 6 \
        --refine --refine_iter 1 \
        $merge_opt \
        --out_prefix "$PREFIX" \
        --calc_acc > /dev/null 2>&1 || true

      # 精准定位 metrics 文件路径
      METRICS_FILE="$(dirname "$PREFIX")/cluster_results/$(basename "$PREFIX").mclust.metrics.txt"

      if [ -f "$METRICS_FILE" ]; then
        # 修复点：使用 awk 精确提取制表符分隔的数值
        final_k=$(awk '$1=="n_clusters" {print $2}' "$METRICS_FILE")
        ari=$(awk '$1=="ARI" {print $2}' "$METRICS_FILE")
        nmi=$(awk '$1=="NMI" {print $2}' "$METRICS_FILE")
        acc=$(awk '$1=="ACC" {print $2}' "$METRICS_FILE")

        [ -z "$final_k" ] && final_k="NA"
        [ -z "$ari" ] && ari="NA"
        [ -z "$nmi" ] && nmi="NA"
        [ -z "$acc" ] && acc="NA"

        printf "    -> Result: Final_K=%s, ARI=%.4f, NMI=%.4f\n" "$final_k" "$ari" "$nmi"
        echo -e "${pca}\t${k}\t${merge_tag}\t${final_k}\t${ari}\t${nmi}\t${acc}" >> "$LOG_FILE"
      else
        echo "    [!] 失败或未生成指标文件: $METRICS_FILE"
      fi

      current=$((current + 1))
    done
  done
done

echo "=================================================================="
echo "扫描完成！按 ARI 降序排列的前 10 名："
{
  head -n 1 "$LOG_FILE"
  tail -n +2 "$LOG_FILE" | sort -t$'\t' -k5,5gr | head -n 10
} | column -t -s $'\t'