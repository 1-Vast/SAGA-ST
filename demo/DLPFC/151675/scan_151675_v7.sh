#!/bin/bash
set -uo pipefail

echo "=================================================================="
echo "🚀 151675 破壁计划 V7 (目标 ARI > 0.6)"
echo "策略: 24组高阶特征重构 + 黄金聚类模板 (锁定保护 55 Spot 薄层)"
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

H5_PATH="/root/autodl-tmp/DLPFC/151675_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/151675_tuning_v7"
mkdir -p "$OUT_DIR/cluster_results"

SUMMARY_LOG="$OUT_DIR/tuning_summary_v7.log"
echo "=== DLPFC 151675 Hyperparameter Tuning V7 ===" > "$SUMMARY_LOG"
echo -e "K_KNN\tALPHA\tL_RECON\tMASK\tC_TYPE\tN_CLUST\tARI\tNMI" >> "$SUMMARY_LOG"

# ==========================================
# 🔎 V7 训练搜索空间 (3*2*2*2 = 24组训练)
# ==========================================
SPATIAL_K_LIST=(10 12 14)          # [扩展] 尝试更大的感受野 14
ALPHA_LIST=(0.5 0.7)               # [核心] 0.7 强势保护薄层节点
LAMBDA_RECON_LIST=(0.1 0.15)       # 维持高重构，逼迫模型记住结构
MASK_RATIO_LIST=(0.1 0.2)          # 低强度扰动
GAMMA=2.0                          # 固定最佳 Gamma
HARD_R=0.6                         # 固定最佳负样本比例

total_runs=$(( ${#SPATIAL_K_LIST[@]} * ${#ALPHA_LIST[@]} * ${#LAMBDA_RECON_LIST[@]} * ${#MASK_RATIO_LIST[@]} ))
current_run=1

echo "预计总训练次数: $total_runs"

for k_knn in "${SPATIAL_K_LIST[@]}"; do
  for alpha in "${ALPHA_LIST[@]}"; do
    for l_recon in "${LAMBDA_RECON_LIST[@]}"; do
      for mask_r in "${MASK_RATIO_LIST[@]}"; do
        
        PREFIX_NAME="run_k${k_knn}_a${alpha}_lr${l_recon}_mr${mask_r}"
        CURRENT_PREFIX="$OUT_DIR/$PREFIX_NAME"
        
        echo "------------------------------------------------------------------"
        echo ">>> [$current_run/$total_runs] Train -> k=$k_knn, alpha=$alpha, l_recon=$l_recon, mask=$mask_r"
        
        # 1. 运行 main.py
        python -m model.main \
          --h5 "$H5_PATH" \
          --out_prefix "$CURRENT_PREFIX" \
          --graph_model KNN --k "$k_knn" \
          --use_scanpy_workflow --pca_comps 64 \
          --dim 64 --K 2 --embed_agg concat --hidden 512 \
          --alpha "$alpha" --topN 50 \
          --lambda_recon "$l_recon" \
          --mask_ratio_feat "$mask_r" \
          --epochs 1200 \
          --pos_per_epoch 20000 \
          --layer_aware --no_layer_fallback \
          --pseudo_layer_bins 7 --pseudo_layer_knn 20 \
          --neg_layer_margin 2 --layer_gamma "$GAMMA" \
          --neg_hard_ratio "$HARD_R" --neg_oversample 8 \
          --normal_aware \
          --normal_knn "$k_knn" --normal_margin 1.0 --normal_gamma 2.0 \
          --activation prelu --scheduler \
          --lr 5e-4 --weight_decay 1e-4 \
          --seed 42 --device cuda > /dev/null 2>&1 || true
        
        NPZ_FILE="${CURRENT_PREFIX}.augK2_d64_for_cluster.npz"

        if [ ! -f "$NPZ_FILE" ]; then
          echo "    [!] 训练失败，跳过聚类。"
          current_run=$((current_run + 1))
          continue
        fi

        # 2. 聚类端：使用证明过能拿高分的黄金策略，卡死 min_cluster_size=40
        CLUST_CONFIGS=(
          "Gold_A|8|20|2"  # 你的 0.5959 配置 (smooth_k=8, refine_k=20, iter=2)
          "Gold_B|6|20|1"  # 稍微保守一点的版本，防止 iter=2 侵蚀过度
        )

        for config in "${CLUST_CONFIGS[@]}"; do
          IFS="|" read -r c_name c_smooth c_refine c_iter <<< "$config"
          METRICS_FILE="$(dirname "$CURRENT_PREFIX")/cluster_results/$(basename "$CURRENT_PREFIX").${c_name}.metrics.txt"
          
          python -m model.cluster \
            --npz "$NPZ_FILE" \
            --h5 "$H5_PATH" \
            --label_key sce.layer_guess \
            --method mclust \
            --use_rep emb \
            --pca_dim 30 \
            --n_clusters 7 \
            --smooth \
            --smooth_k "$c_smooth" \
            --refine \
            --refine_k "$c_refine" \
            --refine_iter "$c_iter" \
            --merge_small \
            --min_cluster_size 40 \
            --calc_acc > "$METRICS_FILE" 2>&1 || true
            
          # 3. 提取指标
          if [ -f "$METRICS_FILE" ]; then
            n_clust=$(awk '$1=="Number" && $2=="of" && $3=="clusters:" {print $4}' "$METRICS_FILE" | head -n 1) || n_clust="NA"
            if [ -z "$n_clust" ]; then n_clust=$(awk '$1=="n_clusters" {print $2}' "$METRICS_FILE"); fi
            
            ari=$(awk '$1=="ARI:" {print $2}' "$METRICS_FILE" | head -n 1) || ari="0"
            if [ -z "$ari" ]; then ari=$(awk '$1=="ARI" {print $2}' "$METRICS_FILE"); fi
            
            nmi=$(awk '$1=="NMI:" {print $2}' "$METRICS_FILE" | head -n 1) || nmi="0"
            if [ -z "$nmi" ]; then nmi=$(awk '$1=="NMI" {print $2}' "$METRICS_FILE"); fi
            
            is_high=$(awk -v a="$ari" 'BEGIN {print (a >= 0.60) ? 1 : 0}')
            if [ "$is_high" -eq 1 ]; then
                printf "      🔥 [%s] ARI=\033[32m%.4f\033[0m, NMI=%.4f, K=%s\n" "$c_name" "$ari" "$nmi" "$n_clust"
            else
                printf "      Result: [%s] -> ARI=%.4f, NMI=%.4f, K=%s\n" "$c_name" "$ari" "$nmi" "$n_clust"
            fi
            
            echo -e "${k_knn}\t${alpha}\t${l_recon}\t${mask_r}\t${c_name}\t${n_clust}\t${ari}\t${nmi}" >> "$SUMMARY_LOG"
          fi
          rm -f "$METRICS_FILE"
        done
        
        # 4. 训练结束，只留精华 NPZ
        rm -f "${CURRENT_PREFIX}"*.h5ad
        current_run=$((current_run + 1))
      done
    done
  done
done

echo "------------------------------------------------------------------"
echo "✅ Tuning V7 finished! Check the results in: $SUMMARY_LOG"
sort -t$'\t' -k7 -n -r "$SUMMARY_LOG" | head -n 11
echo "=================================================================="

# root@autodl-container-36fd48b95a-979ea069:~/autodl-tmp/Spatial-main# bash /root/autodl-tmp/Spatial-main/scan_151675_v7.sh
# ==================================================================
# 🚀 151675 破壁计划 V7 (目标 ARI > 0.6)
# 策略: 24组高阶特征重构 + 黄金聚类模板 (锁定保护 55 Spot 薄层)
# ==================================================================
# 预计总训练次数: 24
# ------------------------------------------------------------------
# >>> [1/24] Train -> k=10, alpha=0.5, l_recon=0.1, mask=0.1
#       Result: [Gold_A] -> ARI=0.4503, NMI=0.5810, K=6
#       Result: [Gold_B] -> ARI=0.3889, NMI=0.5487, K=6
# ------------------------------------------------------------------
# >>> [2/24] Train -> k=10, alpha=0.5, l_recon=0.1, mask=0.2
#       Result: [Gold_A] -> ARI=0.4149, NMI=0.5613, K=6
#       Result: [Gold_B] -> ARI=0.4858, NMI=0.6250, K=6
# ------------------------------------------------------------------
# >>> [3/24] Train -> k=10, alpha=0.5, l_recon=0.15, mask=0.1
#       Result: [Gold_A] -> ARI=0.4943, NMI=0.6260, K=6
#       Result: [Gold_B] -> ARI=0.4383, NMI=0.6127, K=6
# ------------------------------------------------------------------
# >>> [4/24] Train -> k=10, alpha=0.5, l_recon=0.15, mask=0.2
#       Result: [Gold_A] -> ARI=0.4599, NMI=0.6021, K=6
#       Result: [Gold_B] -> ARI=0.4774, NMI=0.6127, K=6
# ------------------------------------------------------------------
# >>> [5/24] Train -> k=10, alpha=0.7, l_recon=0.1, mask=0.1
#       Result: [Gold_A] -> ARI=0.3959, NMI=0.5735, K=6
#       Result: [Gold_B] -> ARI=0.5024, NMI=0.6273, K=6
# ------------------------------------------------------------------
# >>> [6/24] Train -> k=10, alpha=0.7, l_recon=0.1, mask=0.2
#       Result: [Gold_A] -> ARI=0.4821, NMI=0.6329, K=6
#       Result: [Gold_B] -> ARI=0.4697, NMI=0.6077, K=6
# ------------------------------------------------------------------
# >>> [7/24] Train -> k=10, alpha=0.7, l_recon=0.15, mask=0.1
#       Result: [Gold_A] -> ARI=0.3771, NMI=0.5291, K=6
#       Result: [Gold_B] -> ARI=0.4804, NMI=0.6108, K=6
# ------------------------------------------------------------------
# >>> [8/24] Train -> k=10, alpha=0.7, l_recon=0.15, mask=0.2
#       Result: [Gold_A] -> ARI=0.4292, NMI=0.6032, K=6
#       Result: [Gold_B] -> ARI=0.5368, NMI=0.6550, K=6
# ------------------------------------------------------------------
# >>> [9/24] Train -> k=12, alpha=0.5, l_recon=0.1, mask=0.1
#       Result: [Gold_A] -> ARI=0.4308, NMI=0.5907, K=6
#       Result: [Gold_B] -> ARI=0.4562, NMI=0.5914, K=5
# ------------------------------------------------------------------
# >>> [10/24] Train -> k=12, alpha=0.5, l_recon=0.1, mask=0.2
#       Result: [Gold_A] -> ARI=0.3972, NMI=0.5609, K=6
#       Result: [Gold_B] -> ARI=0.4348, NMI=0.5906, K=6
# ------------------------------------------------------------------
# >>> [11/24] Train -> k=12, alpha=0.5, l_recon=0.15, mask=0.1
#       Result: [Gold_A] -> ARI=0.3849, NMI=0.5494, K=6
#       Result: [Gold_B] -> ARI=0.4885, NMI=0.6257, K=6
# ------------------------------------------------------------------
# >>> [12/24] Train -> k=12, alpha=0.5, l_recon=0.15, mask=0.2
#       Result: [Gold_A] -> ARI=0.5174, NMI=0.6621, K=6
#       Result: [Gold_B] -> ARI=0.4412, NMI=0.5937, K=5
# ------------------------------------------------------------------
# >>> [13/24] Train -> k=12, alpha=0.7, l_recon=0.1, mask=0.1
#       Result: [Gold_A] -> ARI=0.4268, NMI=0.6116, K=6
#       Result: [Gold_B] -> ARI=0.4531, NMI=0.5845, K=6
# ------------------------------------------------------------------
# >>> [14/24] Train -> k=12, alpha=0.7, l_recon=0.1, mask=0.2
#       Result: [Gold_A] -> ARI=0.3839, NMI=0.5318, K=6
#       🔥 [Gold_B] ARI=0.6078, NMI=0.6980, K=6
# ------------------------------------------------------------------
# >>> [15/24] Train -> k=12, alpha=0.7, l_recon=0.15, mask=0.1
#       Result: [Gold_A] -> ARI=0.4216, NMI=0.6303, K=6
#       Result: [Gold_B] -> ARI=0.4178, NMI=0.6066, K=6
# ------------------------------------------------------------------
# >>> [16/24] Train -> k=12, alpha=0.7, l_recon=0.15, mask=0.2
#       Result: [Gold_A] -> ARI=0.5268, NMI=0.6399, K=6
#       Result: [Gold_B] -> ARI=0.5353, NMI=0.6193, K=6
# ------------------------------------------------------------------
# >>> [17/24] Train -> k=14, alpha=0.5, l_recon=0.1, mask=0.1
#       Result: [Gold_A] -> ARI=0.3926, NMI=0.5466, K=6
#       Result: [Gold_B] -> ARI=0.4597, NMI=0.5981, K=6
# ------------------------------------------------------------------
# >>> [18/24] Train -> k=14, alpha=0.5, l_recon=0.1, mask=0.2
#       Result: [Gold_A] -> ARI=0.4038, NMI=0.5993, K=6
#       Result: [Gold_B] -> ARI=0.4923, NMI=0.6137, K=6
# ------------------------------------------------------------------
# >>> [19/24] Train -> k=14, alpha=0.5, l_recon=0.15, mask=0.1
#       Result: [Gold_A] -> ARI=0.5075, NMI=0.6110, K=6
#       Result: [Gold_B] -> ARI=0.4665, NMI=0.6006, K=6
# ------------------------------------------------------------------
# >>> [20/24] Train -> k=14, alpha=0.5, l_recon=0.15, mask=0.2
#       Result: [Gold_A] -> ARI=0.3682, NMI=0.5305, K=6
#       Result: [Gold_B] -> ARI=0.4569, NMI=0.5809, K=6
# ------------------------------------------------------------------
# >>> [21/24] Train -> k=14, alpha=0.7, l_recon=0.1, mask=0.1
#       Result: [Gold_A] -> ARI=0.3998, NMI=0.5445, K=6
#       Result: [Gold_B] -> ARI=0.5566, NMI=0.6427, K=6
# ------------------------------------------------------------------
# >>> [22/24] Train -> k=14, alpha=0.7, l_recon=0.1, mask=0.2
#       Result: [Gold_A] -> ARI=0.5064, NMI=0.6232, K=6
#       Result: [Gold_B] -> ARI=0.5028, NMI=0.6319, K=6
# ------------------------------------------------------------------
# >>> [23/24] Train -> k=14, alpha=0.7, l_recon=0.15, mask=0.1
#       Result: [Gold_A] -> ARI=0.4770, NMI=0.6066, K=6
#       Result: [Gold_B] -> ARI=0.5239, NMI=0.6410, K=6
# ------------------------------------------------------------------
# >>> [24/24] Train -> k=14, alpha=0.7, l_recon=0.15, mask=0.2
#       Result: [Gold_A] -> ARI=0.5197, NMI=0.6166, K=6
#       Result: [Gold_B] -> ARI=0.4931, NMI=0.6317, K=6
# ------------------------------------------------------------------
# ✅ Tuning V7 finished! Check the results in: /root/autodl-tmp/DLPFC/151675_tuning_v7/tuning_summary_v7.log
# 12      0.7     0.1     0.2     Gold_B  6       0.6078  0.6980
# 14      0.7     0.1     0.1     Gold_B  6       0.5566  0.6427
# 10      0.7     0.15    0.2     Gold_B  6       0.5368  0.6550
# 12      0.7     0.15    0.2     Gold_B  6       0.5353  0.6193
# 12      0.7     0.15    0.2     Gold_A  6       0.5268  0.6399
# 14      0.7     0.15    0.1     Gold_B  6       0.5239  0.6410
# 14      0.7     0.15    0.2     Gold_A  6       0.5197  0.6166
# 12      0.5     0.15    0.2     Gold_A  6       0.5174  0.6621
# 14      0.5     0.15    0.1     Gold_A  6       0.5075  0.6110
# 14      0.7     0.1     0.2     Gold_A  6       0.5064  0.6232
# 14      0.7     0.1     0.2     Gold_B  6       0.5028  0.6319
# ==================================================================
# root@autodl-container-36fd48b95a-979ea069:~/autodl-tmp/Spatial-main# 