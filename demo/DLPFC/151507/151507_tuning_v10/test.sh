#!/bin/bash
set -uo pipefail

echo "=================================================================="
echo "🚀 151507 破壁计划 V10 (全面转向 Robust Consensus 聚类)"
echo "策略: 吸纳 T1/T5 黄金配置，利用混合图聚类攻克 Layer 1 极小类"
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

H5_PATH="/root/autodl-tmp/DLPFC/151507_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/151507_tuning_v10"
mkdir -p "$OUT_DIR/cluster_results"

SUMMARY_LOG="$OUT_DIR/tuning_summary_v10.log"
echo "=== DLPFC 151507 Hyperparameter Tuning V10 ===" > "$SUMMARY_LOG"
echo -e "K_KNN\tALPHA\tL_RECON\tMASK\tC_TYPE\tN_CLUST\tARI\tNMI" >> "$SUMMARY_LOG"

# ==========================================
# 🔎 V10 训练空间 (1*3*3*2 = 18组训练，贴合 T1/T5 锚点)
# ==========================================
SPATIAL_K_LIST=(10)                # 锁定最佳空间邻居
ALPHA_LIST=(0.45 0.50 0.55)        # 围绕 T1 的 0.50 微调
LAMBDA_RECON_LIST=(0.05 0.08 0.10) # 包含 T1 的 0.08
MASK_RATIO_LIST=(0.08 0.10)        # 包含 T1 的 0.08
GAMMA=2.0                          
HARD_R=0.60                        

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
        
        # 1. 运行 main.py (同步了 T1 的正常向量和权重衰减优化)
        python -m model.main \
          --h5 "$H5_PATH" \
          --out_prefix "$CURRENT_PREFIX" \
          --graph_model KNN --k "$k_knn" \
          --use_scanpy_workflow --pca_comps 64 \
          --dim 64 --K 2 --embed_agg concat --hidden 512 \
          --alpha "$alpha" --topN 50 \
          --lambda_recon "$l_recon" \
          --mask_ratio_feat "$mask_r" \
          --epochs 1000 \
          --pos_per_epoch 18000 \
          --layer_aware --no_layer_fallback \
          --pseudo_layer_bins 7 --pseudo_layer_knn 20 \
          --neg_layer_margin 2 --layer_gamma "$GAMMA" \
          --neg_hard_ratio "$HARD_R" --neg_oversample 8 \
          --normal_aware \
          --normal_knn 10 --normal_margin 0.8 --normal_gamma 1.8 \
          --activation prelu --scheduler \
          --lr 5e-4 --weight_decay 5e-5 \
          --seed 42 --device cuda > /dev/null 2>&1 || true
        
        NPZ_FILE="${CURRENT_PREFIX}.augK2_d64_for_cluster.npz"

        if [ ! -f "$NPZ_FILE" ]; then
          echo "    [!] 训练失败，跳过聚类。"
          current_run=$((current_run + 1))
          continue
        fi

        # ==========================================
        # 聚类配置梯队测试 (Robust Consensus)
        # 格式: Name | pca_dim | w_spa | w_emb | refine_k
        # ==========================================
        CLUST_CONFIGS=(
          "R_P24_W30_r10|24|0.30|0.70|10"   # T1/T5 黄金锚点
          "R_P20_W30_r8|20|0.30|0.70|8"     # 降维 + 降平滑 (防吞噬)
          "R_P24_W20_r10|24|0.20|0.80|10"   # 降低空间图权重，更依赖表达
          "R_P24_W40_r10|24|0.40|0.60|10"   # 提高空间图权重，使分层更平滑
        )

        for config in "${CLUST_CONFIGS[@]}"; do
          IFS="|" read -r c_name c_pca c_wspa c_wemb c_refine <<< "$config"
          METRICS_FILE="$(dirname "$CURRENT_PREFIX")/cluster_results/$(basename "$CURRENT_PREFIX").${c_name}.metrics.txt"
          
          CMD=(python -m model.cluster \
            --npz "$NPZ_FILE" \
            --h5 "$H5_PATH" \
            --label_key sce.layer_guess \
            --method robust \
            --use_rep emb \
            --pca_dim "$c_pca" \
            --n_clusters 7 \
            --knn_k 10 \
            --resolution 0.68 \
            --robust_res_list "0.62,0.66,0.68,0.72,0.76" \
            --w_spa "$c_wspa" \
            --w_emb "$c_wemb" \
            --robust_seeds 9 \
            --robust_smooth_iter 0 \
            --refine_k "$c_refine" \
            --refine_iter 1 \
            --merge_small --min_cluster_size 10 \
            --calc_acc)
            
          "${CMD[@]}" > "$METRICS_FILE" 2>&1 || true
            
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
        
        # 4. 训练结束，留存精华 NPZ
        rm -f "${CURRENT_PREFIX}"*.h5ad
        current_run=$((current_run + 1))
      done
    done
  done
done

echo "------------------------------------------------------------------"
echo "✅ Tuning V10 finished! Check the results in: $SUMMARY_LOG"
sort -t$'\t' -k7 -n -r "$SUMMARY_LOG" | head -n 11
echo "=================================================================="