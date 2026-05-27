#!/bin/bash
set -uo pipefail

echo "=================================================================="
echo "🚀 151507 破壁计划 V13 (锁定黄金参数，微操登顶 ARI > 0.62)"
echo "策略: 绝对锁定 PCA=24, W_SPA=0.40, 极限微调 Resolution 与 Refine_K"
echo "=================================================================="

unset OMP_NUM_THREADS || true
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# 核心路径配置 (依然使用你那份 0.6066/0.6115 的神仙特征文件)
NPZ_FILE="/root/autodl-tmp/DLPFC/151507_tuning_v10/run_k10_a0.55_lr0.10_mr0.08.augK2_d64_for_cluster.npz"
H5_FILE="/root/autodl-tmp/DLPFC/151507_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/151507_cluster_v13"
mkdir -p "$OUT_DIR/cluster_results"

SUMMARY_LOG="$OUT_DIR/cluster_v13_summary.log"
echo "=== DLPFC 151507 Cluster Micro-Tuning V13 ===" > "$SUMMARY_LOG"
echo -e "RES_BASE\tREF_K\tREF_IT\tN_CLUST\tARI\tNMI\tACC" >> "$SUMMARY_LOG"

# ==========================================
# 🔎 V13 毫米级微操空间 (4 * 5 * 2 = 40 种组合)
# 绝对锁死: PCA=24, W_SPA=0.40, W_EMB=0.60
# ==========================================
PCA=24
W_SPA=0.40
W_EMB=0.60

# 微调分辨率
RES_LIST=(0.66 0.67 0.68 0.69)
# 突破原来的 12，向 13, 14, 15 试探更广阔的平滑视野
REFINE_K_LIST=(11 12 13 14 15)
# 迭代次数
REFINE_ITER_LIST=(2 3)

total_runs=$(( ${#RES_LIST[@]} * ${#REFINE_K_LIST[@]} * ${#REFINE_ITER_LIST[@]} ))
current_run=1

echo "预计微操次数: $total_runs (纯聚类，极速完成)"
echo "------------------------------------------------------------------"

for res in "${RES_LIST[@]}"; do
  # 动态生成共识分辨率列表 (例如 res=0.68 -> 0.62,0.66,0.68,0.72)
  res_minus_06=$(awk -v r="$res" 'BEGIN {printf "%.2f", r-0.06}')
  res_minus_02=$(awk -v r="$res" 'BEGIN {printf "%.2f", r-0.02}')
  res_plus_04=$(awk -v r="$res" 'BEGIN {printf "%.2f", r+0.04}')
  res_list_str="${res_minus_06},${res_minus_02},${res},${res_plus_04}"

  for ref_k in "${REFINE_K_LIST[@]}"; do
    for ref_iter in "${REFINE_ITER_LIST[@]}"; do
      
      RUN_NAME="Res${res}_rK${ref_k}_rI${ref_iter}"
      CURRENT_PREFIX="$OUT_DIR/$RUN_NAME"
      METRICS_FILE="$OUT_DIR/cluster_results/${RUN_NAME}.robust.metrics.txt"
      
      python -m model.cluster \
        --npz "$NPZ_FILE" \
        --h5 "$H5_FILE" \
        --label_key "sce.layer_guess" \
        --method robust \
        --use_rep emb \
        --pca_dim "$PCA" \
        --n_clusters 7 \
        --knn_k 10 \
        --resolution "$res" \
        --robust_res_list "$res_list_str" \
        --w_spa "$W_SPA" \
        --w_emb "$W_EMB" \
        --robust_seeds 9 \
        --robust_smooth_iter 0 \
        --refine_k "$ref_k" \
        --refine_iter "$ref_iter" \
        --merge_small --min_cluster_size 10 \
        --out_prefix "$CURRENT_PREFIX" \
        --calc_acc > /dev/null 2>&1 || true
        
      if [ -f "$METRICS_FILE" ]; then
        n_clust=$(awk '$1=="Number" && $2=="of" && $3=="clusters:" {print $4}' "$METRICS_FILE" | head -n 1) || n_clust="NA"
        if [ -z "$n_clust" ]; then n_clust=$(awk '$1=="n_clusters" {print $2}' "$METRICS_FILE"); fi
        
        ari=$(awk '$1=="ARI:" {print $2}' "$METRICS_FILE" | head -n 1) || ari="0"
        if [ -z "$ari" ]; then ari=$(awk '$1=="ARI" {print $2}' "$METRICS_FILE"); fi
        
        nmi=$(awk '$1=="NMI:" {print $2}' "$METRICS_FILE" | head -n 1) || nmi="0"
        if [ -z "$nmi" ]; then nmi=$(awk '$1=="NMI" {print $2}' "$METRICS_FILE"); fi
        
        acc=$(awk '$1=="ACC:" {print $2}' "$METRICS_FILE" | head -n 1) || acc="0"
        if [ -z "$acc" ]; then acc=$(awk '$1=="ACC" {print $2}' "$METRICS_FILE"); fi
        
        is_god=$(awk -v a="$ari" 'BEGIN {print (a >= 0.62) ? 1 : 0}')
        if [ "$is_god" -eq 1 ]; then
            printf "[%2d/%d] 👑登顶0.62! [%s] -> K=%s, ARI=\033[1;33m%.4f\033[0m, NMI=%.4f\n" "$current_run" "$total_runs" "$RUN_NAME" "$n_clust" "$ari" "$nmi"
        else
            is_high=$(awk -v a="$ari" 'BEGIN {print (a >= 0.61) ? 1 : 0}')
            if [ "$is_high" -eq 1 ]; then
                printf "[%2d/%d] 🔥无限逼近! [%s] -> K=%s, ARI=\033[32m%.4f\033[0m, NMI=%.4f\n" "$current_run" "$total_runs" "$RUN_NAME" "$n_clust" "$ari" "$nmi"
            else
                printf "[%2d/%d] 尝试中... [%s] -> K=%s, ARI=%.4f\n" "$current_run" "$total_runs" "$RUN_NAME" "$n_clust" "$ari"
            fi
        fi
        
        echo -e "${res}\t${ref_k}\t${ref_iter}\t${n_clust}\t${ari}\t${nmi}\t${acc}" >> "$SUMMARY_LOG"
        
        # 只保留 ARI >= 0.61 且 K=7 的高质量结果图
        is_keep=$(awk -v a="$ari" -v k="$n_clust" 'BEGIN {print (a >= 0.61 && k == 7) ? 1 : 0}')
        if [ "$is_keep" -eq 0 ]; then
            rm -f "$OUT_DIR/cluster_results/${RUN_NAME}"*
        fi
      else
        printf "[%2d/%d] ❌ 运行失败 [%s]\n" "$current_run" "$total_runs" "$RUN_NAME"
      fi
      
      current_run=$((current_run + 1))
    done
  done
done

echo "------------------------------------------------------------------"
echo "✅ 微操扫荡完成！查看巅峰排行榜 (Top 10):"
echo "日志文件: $SUMMARY_LOG"
echo "------------------------------------------------------------------"
sort -t$'\t' -k5 -n -r "$SUMMARY_LOG" | head -n 11
echo "=================================================================="

