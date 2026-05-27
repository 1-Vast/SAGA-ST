#!/bin/bash
set -uo pipefail

echo "=================================================================="
echo "🚀 151673 跨切片平移测试 (复用 151674 高分搜索空间与聚类)"
echo "策略: 27 组训练, normal_aware 体系, smooth_k=6 聚类平滑"
echo "=================================================================="

# 设置严格的线程和确定性环境
unset OMP_NUM_THREADS || true
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export NUMBA_THREADING_LAYER=workqueue
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

# ==========================================
# 🎯 目标切片：151673
# ==========================================
H5_PATH="/root/autodl-tmp/DLPFC/151673_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/151673_tuning_ref151674"
mkdir -p "$OUT_DIR/cluster_results"

# 汇总日志文件
SUMMARY_LOG="$OUT_DIR/tuning_summary.log"
echo "=== DLPFC 151673 Hyperparameter Tuning (Ref: 151674) ===" > "$SUMMARY_LOG"
echo -e "K_KNN\tGAMMA\tHARD_R\tPCA_DIM\tN_CLUST\tARI\tNMI" >> "$SUMMARY_LOG"

# ==========================================
# 搜索空间 (原汁原味保留 151674 的设定)
# ==========================================
SPATIAL_K_LIST=(10 12 15)
GAMMA_LIST=(2.0 3.0 4.0)
HARD_RATIO_LIST=(0.5 0.6 0.7)
CLUSTER_PCA_LIST=(30 50)

total_runs=$(( ${#SPATIAL_K_LIST[@]} * ${#GAMMA_LIST[@]} * ${#HARD_RATIO_LIST[@]} ))
current_run=1

echo "预计总训练次数: $total_runs"

# 开始网格搜索循环
for k_knn in "${SPATIAL_K_LIST[@]}"; do
  for gamma in "${GAMMA_LIST[@]}"; do
    for hard_r in "${HARD_RATIO_LIST[@]}"; do
      
      PREFIX_NAME="run_k${k_knn}_g${gamma}_hr${hard_r}"
      CURRENT_PREFIX="$OUT_DIR/$PREFIX_NAME"
      
      echo "------------------------------------------------------------------"
      echo ">>> [$current_run/$total_runs] Training -> k=$k_knn, gamma=$gamma, hard_ratio=$hard_r ..."
      
      # 1. 运行 main.py (完全复用 151674 脚本中的所有复杂参数设定)
      python -m model.main \
        --h5 "$H5_PATH" \
        --out_prefix "$CURRENT_PREFIX" \
        --graph_model KNN --k "$k_knn" \
        --use_scanpy_workflow --pca_comps 64 \
        --dim 64 --K 2 --embed_agg concat --hidden 512 \
        --alpha 0.5 --topN 50 \
        --lambda_recon 0.05 \
        --mask_ratio_feat 0.2 \
        --epochs 1200 \
        --pos_per_epoch 20000 \
        --layer_aware --no_layer_fallback \
        --pseudo_layer_bins 7 --pseudo_layer_knn 20 \
        --neg_layer_margin 2 --layer_gamma "$gamma" \
        --neg_hard_ratio "$hard_r" --neg_oversample 8 \
        --normal_aware \
        --normal_knn 15 --normal_margin 1.0 --normal_gamma 2.0 \
        --activation prelu --scheduler \
        --lr 5e-4 --weight_decay 1e-4 \
        --seed 42 --device cuda > /dev/null 2>&1 || true
      
      NPZ_FILE="${CURRENT_PREFIX}.augK2_d64_for_cluster.npz"

      if [ ! -f "$NPZ_FILE" ]; then
        echo "    [!] 训练失败或显存溢出，跳过聚类。"
        current_run=$((current_run + 1))
        continue
      fi

      # 2. 运行 cluster.py (复用 smooth_k 6 和 refine 策略)
      for pca_dim in "${CLUSTER_PCA_LIST[@]}"; do
        
        METRICS_FILE="$(dirname "$CURRENT_PREFIX")/cluster_results/$(basename "$CURRENT_PREFIX").augK2_d64_for_cluster.mclust.metrics.txt"
        
        python -m model.cluster \
          --npz "$NPZ_FILE" \
          --h5 "$H5_PATH" \
          --label_key sce.layer_guess \
          --method mclust \
          --use_rep emb \
          --pca_dim "$pca_dim" \
          --n_clusters 7 \
          --smooth \
          --smooth_k 6 \
          --refine \
          --refine_iter 1 \
          --calc_acc > /dev/null 2>&1 || true
          
        # 3. 提取指标并安全地判断分数 (使用 awk)
        if [ -f "$METRICS_FILE" ]; then
          n_clust=$(awk '$1=="n_clusters" {print $2}' "$METRICS_FILE") || n_clust="NA"
          ari=$(awk '$1=="ARI" {print $2}' "$METRICS_FILE") || ari="0"
          nmi=$(awk '$1=="NMI" {print $2}' "$METRICS_FILE") || nmi="0"
          
          is_high=$(awk -v a="$ari" 'BEGIN {print (a >= 0.55) ? 1 : 0}')
          if [ "$is_high" -eq 1 ]; then
              printf "       🔥 [HIGH SCORE!] PCA=%s -> ARI=\033[31m%.4f\033[0m, NMI=%.4f\n" "$pca_dim" "$ari" "$nmi"
          else
              printf "       Result: PCA=%s -> ARI=%.4f, NMI=%.4f\n" "$pca_dim" "$ari" "$nmi"
          fi
          
          echo -e "${k_knn}\t${gamma}\t${hard_r}\t${pca_dim}\t${n_clust}\t${ari}\t${nmi}" >> "$SUMMARY_LOG"
        fi
      done
      
      # 清理当次训练生成的冗余 h5ad 文件，只保留宝贵的 npz 提纯特征
      rm -f "${CURRENT_PREFIX}"*.h5ad
      
      current_run=$((current_run + 1))
  done
done
done

echo "------------------------------------------------------------------"
echo "✅ Tuning finished! Check the results in: $SUMMARY_LOG"
echo "Top 5 configurations sorted by ARI:"
sort -t$'\t' -k6 -n -r "$SUMMARY_LOG" | head -n 6
echo "=================================================================="


# root@autodl-container-85b6438982-162544e8:~/autodl-tmp/Spatial-main# bash scan.sh
# ==================================================================
# 🚀 151673 跨切片平移测试 (复用 151674 高分搜索空间与聚类)
# 策略: 27 组训练, normal_aware 体系, smooth_k=6 聚类平滑
# ==================================================================
# 预计总训练次数: 27
# ------------------------------------------------------------------
# >>> [1/27] Training -> k=10, gamma=2.0, hard_ratio=0.5 ...
#        Result: PCA=30 -> ARI=0.3985, NMI=0.5648
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5851, NMI=0.6456
# ------------------------------------------------------------------
# >>> [2/27] Training -> k=10, gamma=2.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.5443, NMI=0.6296
#        Result: PCA=50 -> ARI=0.4087, NMI=0.5741
# ------------------------------------------------------------------
# >>> [3/27] Training -> k=10, gamma=2.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.4903, NMI=0.5819
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5564, NMI=0.6505
# ------------------------------------------------------------------
# >>> [4/27] Training -> k=10, gamma=3.0, hard_ratio=0.5 ...
#        Result: PCA=30 -> ARI=0.4964, NMI=0.6359
#        Result: PCA=50 -> ARI=0.3549, NMI=0.5274
# ------------------------------------------------------------------
# >>> [5/27] Training -> k=10, gamma=3.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4608, NMI=0.6003
#        Result: PCA=50 -> ARI=0.4847, NMI=0.6297
# ------------------------------------------------------------------
# >>> [6/27] Training -> k=10, gamma=3.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.5259, NMI=0.6211
#        Result: PCA=50 -> ARI=0.4708, NMI=0.6303
# ------------------------------------------------------------------
# >>> [7/27] Training -> k=10, gamma=4.0, hard_ratio=0.5 ...
#        Result: PCA=30 -> ARI=0.5218, NMI=0.6060
#        Result: PCA=50 -> ARI=0.3893, NMI=0.5603
# ------------------------------------------------------------------
# >>> [8/27] Training -> k=10, gamma=4.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4588, NMI=0.5830
#        Result: PCA=50 -> ARI=0.5142, NMI=0.6081
# ------------------------------------------------------------------
# >>> [9/27] Training -> k=10, gamma=4.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.4690, NMI=0.6075
#        Result: PCA=50 -> ARI=0.4192, NMI=0.5787
# ------------------------------------------------------------------
# >>> [10/27] Training -> k=12, gamma=2.0, hard_ratio=0.5 ...
#        Result: PCA=30 -> ARI=0.4401, NMI=0.6057
#        Result: PCA=50 -> ARI=0.4106, NMI=0.5637
# ------------------------------------------------------------------
# >>> [11/27] Training -> k=12, gamma=2.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4169, NMI=0.5951
#        Result: PCA=50 -> ARI=0.3048, NMI=0.4820
# ------------------------------------------------------------------
# >>> [12/27] Training -> k=12, gamma=2.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.4101, NMI=0.5754
#        Result: PCA=50 -> ARI=0.4341, NMI=0.5862
# ------------------------------------------------------------------
# >>> [13/27] Training -> k=12, gamma=3.0, hard_ratio=0.5 ...
#        🔥 [HIGH SCORE!] PCA=30 -> ARI=0.5842, NMI=0.6734
#        Result: PCA=50 -> ARI=0.3886, NMI=0.5573
# ------------------------------------------------------------------
# >>> [14/27] Training -> k=12, gamma=3.0, hard_ratio=0.6 ...
#        🔥 [HIGH SCORE!] PCA=30 -> ARI=0.5637, NMI=0.6382
#        Result: PCA=50 -> ARI=0.5196, NMI=0.6561
# ------------------------------------------------------------------
# >>> [15/27] Training -> k=12, gamma=3.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.4712, NMI=0.6063
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5966, NMI=0.6673
# ------------------------------------------------------------------
# >>> [16/27] Training -> k=12, gamma=4.0, hard_ratio=0.5 ...
#        🔥 [HIGH SCORE!] PCA=30 -> ARI=0.5757, NMI=0.6374
#        Result: PCA=50 -> ARI=0.4404, NMI=0.6201
# ------------------------------------------------------------------
# >>> [17/27] Training -> k=12, gamma=4.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4638, NMI=0.5991
#        Result: PCA=50 -> ARI=0.4112, NMI=0.5770
# ------------------------------------------------------------------
# >>> [18/27] Training -> k=12, gamma=4.0, hard_ratio=0.7 ...
#        🔥 [HIGH SCORE!] PCA=30 -> ARI=0.5652, NMI=0.6543
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5598, NMI=0.6711
# ------------------------------------------------------------------
# >>> [19/27] Training -> k=15, gamma=2.0, hard_ratio=0.5 ...
#        Result: PCA=30 -> ARI=0.4687, NMI=0.5976
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5592, NMI=0.6415
# ------------------------------------------------------------------
# >>> [20/27] Training -> k=15, gamma=2.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4115, NMI=0.5768
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5819, NMI=0.6671
# ------------------------------------------------------------------
# >>> [21/27] Training -> k=15, gamma=2.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.3764, NMI=0.5509
#        Result: PCA=50 -> ARI=0.4901, NMI=0.6390
# ------------------------------------------------------------------
# >>> [22/27] Training -> k=15, gamma=3.0, hard_ratio=0.5 ...
#        🔥 [HIGH SCORE!] PCA=30 -> ARI=0.5532, NMI=0.6416
#        Result: PCA=50 -> ARI=0.5051, NMI=0.6322
# ------------------------------------------------------------------
# >>> [23/27] Training -> k=15, gamma=3.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4662, NMI=0.6025
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.5835, NMI=0.6857
# ------------------------------------------------------------------
# >>> [24/27] Training -> k=15, gamma=3.0, hard_ratio=0.7 ...
#        Result: PCA=30 -> ARI=0.5449, NMI=0.6352
#        Result: PCA=50 -> ARI=0.4687, NMI=0.6238
# ------------------------------------------------------------------
# >>> [25/27] Training -> k=15, gamma=4.0, hard_ratio=0.5 ...
#        Result: PCA=30 -> ARI=0.4575, NMI=0.5928
#        Result: PCA=50 -> ARI=0.5481, NMI=0.6776
# ------------------------------------------------------------------
# >>> [26/27] Training -> k=15, gamma=4.0, hard_ratio=0.6 ...
#        Result: PCA=30 -> ARI=0.4801, NMI=0.5881
#        🔥 [HIGH SCORE!] PCA=50 -> ARI=0.6409, NMI=0.6887
# ------------------------------------------------------------------
# >>> [27/27] Training -> k=15, gamma=4.0, hard_ratio=0.7 ...
#        🔥 [HIGH SCORE!] PCA=30 -> ARI=0.5653, NMI=0.6410
#        Result: PCA=50 -> ARI=0.3680, NMI=0.5364
# ------------------------------------------------------------------
# ✅ Tuning finished! Check the results in: /root/autodl-tmp/DLPFC/151673_tuning_ref151674/tuning_summary.log
# Top 5 configurations sorted by ARI:
# 15      4.0     0.6     50      7       0.6409026748020715      0.6887202538265403
# 12      3.0     0.7     50      7       0.5965825647776528      0.6672903323450119
# 10      2.0     0.5     50      7       0.5850612411244341      0.645629347866316
# 12      3.0     0.5     30      7       0.5842181781950967      0.6734298113198603
# 15      3.0     0.6     50      7       0.5834681465871087      0.6857359560505448
# 15      2.0     0.6     50      7       0.5818679441493617      0.6670797790298139
# ==================================================================
# root@autodl-container-85b6438982-162544e8:~/autodl-tmp/Spatial-main# 