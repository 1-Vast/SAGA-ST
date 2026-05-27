#!/bin/bash

# 设置严格的线程和确定性环境
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export NUMBA_THREADING_LAYER=workqueue
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

# 全局路径配置
H5_PATH="/root/autodl-tmp/DLPFC/151674_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/tuning_results"
mkdir -p "$OUT_DIR"

# 汇总日志文件
SUMMARY_LOG="$OUT_DIR/tuning_summary.log"
echo "=== DLPFC 151674 Hyperparameter Tuning ===" > "$SUMMARY_LOG"
echo -e "K_KNN\tGAMMA\tHARD_R\tPCA_DIM\tN_CLUST\tARI\tNMI" >> "$SUMMARY_LOG"

# ==========================================
# 定义搜索空间 (你可以根据需要增删这里的值)
# ==========================================
# 1. 空间建图的邻居数 (影响局部感受野的大小)
SPATIAL_K_LIST=(10 12 15)

# 2. 层级负样本的推斥力强度 (过大会撕裂同层，过小会导致层边界模糊)
GAMMA_LIST=(2.0 3.0 4.0)

# 3. 困难负样本挖掘比例 (挑选相似度高的负样本比例)
HARD_RATIO_LIST=(0.5 0.6 0.7)

# 4. Mclust 聚类前的 PCA 降维维度 (过滤高频噪声)
CLUSTER_PCA_LIST=(30 50)
# ==========================================

total_runs=$(( ${#SPATIAL_K_LIST[@]} * ${#GAMMA_LIST[@]} * ${#HARD_RATIO_LIST[@]} * ${#CLUSTER_PCA_LIST[@]} ))
current_run=1

echo "Total configurations to test: $total_runs"

# 开始网格搜索循环
for k_knn in "${SPATIAL_K_LIST[@]}"; do
  for gamma in "${GAMMA_LIST[@]}"; do
    for hard_r in "${HARD_RATIO_LIST[@]}"; do
      
      # 为当前参数组合生成唯一的输出前缀
      PREFIX_NAME="run_k${k_knn}_g${gamma}_hr${hard_r}"
      CURRENT_PREFIX="$OUT_DIR/$PREFIX_NAME"
      
      echo "------------------------------------------------------------------"
      echo ">>> [$current_run/$total_runs] Training with k=$k_knn, gamma=$gamma, hard_ratio=$hard_r ..."
      
      # 1. 运行 main.py (训练特征)
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
        --seed 42 --device cuda > /dev/null 2>&1
      
      # 训练产生的 npz 文件名 (注意这里的 augK2_d64 是由于 main.py 的命名逻辑)
      NPZ_FILE="${CURRENT_PREFIX}.augK2_d64_for_cluster.npz"

      if [ ! -f "$NPZ_FILE" ]; then
        echo "Error: Training failed for $PREFIX_NAME. Skipping clustering."
        current_run=$((current_run + 1))
        continue
      fi

      # 2. 运行 cluster.py (对不同的 PCA 维度进行聚类)
      for pca_dim in "${CLUSTER_PCA_LIST[@]}"; do
        echo "   -> Clustering with pca_dim=$pca_dim ..."
        
        # 聚类结果的指标文件会保存在此处 (根据 cluster.py 输出逻辑)
        # 注意输出路径是在 out_prefix 所在目录的 cluster_results 子文件夹下
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
          --calc_acc > /dev/null 2>&1
          
        # 3. 提取指标并写入汇总表
        if [ -f "$METRICS_FILE" ]; then
          n_clust=$(grep "n_clusters" "$METRICS_FILE" | awk '{print $2}')
          ari=$(grep "ARI" "$METRICS_FILE" | awk '{print $2}')
          nmi=$(grep "NMI" "$METRICS_FILE" | awk '{print $2}')
          
          # 格式化输出到控制台和文件
          printf "      Result: n_clusters=%s, ARI=%.4f, NMI=%.4f\n" "$n_clust" "$ari" "$nmi"
          echo -e "${k_knn}\t${gamma}\t${hard_r}\t${pca_dim}\t${n_clust}\t${ari}\t${nmi}" >> "$SUMMARY_LOG"
        else
          echo "      Error: Metrics file not found for pca_dim=$pca_dim"
        fi
      done
      
      current_run=$((current_run + 1))
      
      # 可选：如果你服务器空间有限，可以加上这行删除中间生成的巨型 npz/h5ad 文件
      # rm -f "${CURRENT_PREFIX}"*.npz "${CURRENT_PREFIX}"*.h5ad
  done
done
done

echo "------------------------------------------------------------------"
echo "Tuning finished! Check the results in: $SUMMARY_LOG"
echo "Top 5 configurations sorted by ARI:"
sort -t$'\t' -k6 -n -r "$SUMMARY_LOG" | head -n 6

# root@autodl-container-cb6a4f850d-177138e0:~/autodl-tmp/Spatial-main# bash tune_dlpfc.sh
# Total configurations to test: 54
# ------------------------------------------------------------------
# >>> [1/54] Training with k=10, gamma=2.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5393, NMI=0.6434
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4369, NMI=0.5377
# ------------------------------------------------------------------
# >>> [2/54] Training with k=10, gamma=2.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4557, NMI=0.5862
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4181, NMI=0.5333
# ------------------------------------------------------------------
# >>> [3/54] Training with k=10, gamma=2.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5640, NMI=0.6399
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.3463, NMI=0.5216
# ------------------------------------------------------------------
# >>> [4/54] Training with k=10, gamma=3.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4454, NMI=0.5632
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4720, NMI=0.5680
# ------------------------------------------------------------------
# >>> [5/54] Training with k=10, gamma=3.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5179, NMI=0.6124
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4469, NMI=0.5572
# ------------------------------------------------------------------
# >>> [6/54] Training with k=10, gamma=3.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5576, NMI=0.6119
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4718, NMI=0.5506
# ------------------------------------------------------------------
# >>> [7/54] Training with k=10, gamma=4.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4595, NMI=0.5747
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.3411, NMI=0.4641
# ------------------------------------------------------------------
# >>> [8/54] Training with k=10, gamma=4.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5075, NMI=0.6022
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4389, NMI=0.5632
# ------------------------------------------------------------------
# >>> [9/54] Training with k=10, gamma=4.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5502, NMI=0.6258
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.5805, NMI=0.6277
# ------------------------------------------------------------------
# >>> [10/54] Training with k=12, gamma=2.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5351, NMI=0.6380
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4372, NMI=0.5615
# ------------------------------------------------------------------
# >>> [11/54] Training with k=12, gamma=2.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5485, NMI=0.6142
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4512, NMI=0.5603
# ------------------------------------------------------------------
# >>> [12/54] Training with k=12, gamma=2.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5516, NMI=0.6268
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4685, NMI=0.5240
# ------------------------------------------------------------------
# >>> [13/54] Training with k=12, gamma=3.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5331, NMI=0.6160
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4989, NMI=0.5744
# ------------------------------------------------------------------
# >>> [14/54] Training with k=12, gamma=3.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4820, NMI=0.5859
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.3926, NMI=0.4976
# ------------------------------------------------------------------
# >>> [15/54] Training with k=12, gamma=3.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5625, NMI=0.6412
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4547, NMI=0.5526
# ------------------------------------------------------------------
# >>> [16/54] Training with k=12, gamma=4.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4915, NMI=0.5926
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4230, NMI=0.5475
# ------------------------------------------------------------------
# >>> [17/54] Training with k=12, gamma=4.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4452, NMI=0.5569
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4602, NMI=0.5753
# ------------------------------------------------------------------
# >>> [18/54] Training with k=12, gamma=4.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.3965, NMI=0.5227
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4389, NMI=0.5658
# ------------------------------------------------------------------
# >>> [19/54] Training with k=15, gamma=2.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4277, NMI=0.5378
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4797, NMI=0.5856
# ------------------------------------------------------------------
# >>> [20/54] Training with k=15, gamma=2.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5801, NMI=0.6637
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.3886, NMI=0.5103
# ------------------------------------------------------------------
# >>> [21/54] Training with k=15, gamma=2.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4054, NMI=0.5464
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4284, NMI=0.5447
# ------------------------------------------------------------------
# >>> [22/54] Training with k=15, gamma=3.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5053, NMI=0.5931
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4222, NMI=0.5336
# ------------------------------------------------------------------
# >>> [23/54] Training with k=15, gamma=3.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4466, NMI=0.5619
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.5538, NMI=0.6289
# ------------------------------------------------------------------
# >>> [24/54] Training with k=15, gamma=3.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4634, NMI=0.5683
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.3908, NMI=0.5189
# ------------------------------------------------------------------
# >>> [25/54] Training with k=15, gamma=4.0, hard_ratio=0.5 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4166, NMI=0.5601
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4934, NMI=0.5784
# ------------------------------------------------------------------
# >>> [26/54] Training with k=15, gamma=4.0, hard_ratio=0.6 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.5228, NMI=0.6294
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.4614, NMI=0.5956
# ------------------------------------------------------------------
# >>> [27/54] Training with k=15, gamma=4.0, hard_ratio=0.7 ...
#    -> Clustering with pca_dim=30 ...
#       Result: n_clusters=7, ARI=0.4534, NMI=0.5802
#    -> Clustering with pca_dim=50 ...
#       Result: n_clusters=7, ARI=0.3218, NMI=0.4937
# ------------------------------------------------------------------
# Tuning finished! Check the results in: /root/autodl-tmp/DLPFC/tuning_results/tuning_summary.log
# Top 5 configurations sorted by ARI:
# 10      4.0     0.7     50      7       0.5805477483924364      0.6277215756984139
# 15      2.0     0.6     30      7       0.580072156156936       0.6636790586919896
# 10      2.0     0.7     30      7       0.5639695505750464      0.6399359705068444
# 12      3.0     0.7     30      7       0.5625302016227998      0.641184141581887
# 10      3.0     0.7     30      7       0.5576009659105923      0.6118634158812196
# 15      3.0     0.6     50      7       0.5538106791134918      0.6288898227036676