#!/bin/bash

# 环境变量设置
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export NUMBA_THREADING_LAYER=workqueue
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

# 全局路径变量
H5_INPUT="/root/autodl-tmp/DLPFC/151671_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/search_trials"          # 与您当前目录一致
LOG_FILE="${OUT_DIR}/master_summary.csv"

# 创建输出目录
mkdir -p "$OUT_DIR"

# ==========================================
# 🎯 核心超参数池 (深度优化版)
# ==========================================
ALPHAS=(0.2 0.3 0.35 0.4)
KS=(4 6 8)
LRECONS=(0.0 0.01 0.05)
NEGHARDS=(0.7 0.8 0.85)
NEGADJS=(0.3 0.4 0.5)

# CSV 表头
echo "Trial,Alpha,K_Graph,LambdaRecon,NegHard,NegAdj,Method,Config,ARI,NMI,ACC" > "$LOG_FILE"

echo "🚀 开始执行 50 次超参数自动化搜索 (适配自动 cluster_results 目录)..."
echo "=================================================="

for i in {1..50}; do
    # 随机抽取参数组合
    ALPHA=${ALPHAS[$RANDOM % ${#ALPHAS[@]}]}
    K=${KS[$RANDOM % ${#KS[@]}]}
    LRECON=${LRECONS[$RANDOM % ${#LRECONS[@]}]}
    NEGHARD=${NEGHARDS[$RANDOM % ${#NEGHARDS[@]}]}
    NEGADJ=${NEGADJS[$RANDOM % ${#NEGADJS[@]}]}

    PREFIX="${OUT_DIR}/trial_${i}"

    echo "⏳ [${i}/50] 训练特征 | Alpha=${ALPHA}, KNN=${K}, LRecon=${LRECON}, Hard=${NEGHARD}, Adj=${NEGADJ}"

    # 1. 执行训练 (保存 NPZ 文件)
    python -m model.main \
      --h5 "$H5_INPUT" \
      --out_prefix "$PREFIX" \
      --graph_model KNN --k $K \
      --use_scanpy_workflow --pca_comps 64 \
      --dim 64 --K 2 --hidden 512 --embed_agg mean \
      --alpha $ALPHA --topN 50 \
      --lambda_recon $LRECON --mask_ratio_feat 0.2 \
      --epochs 1200 --pos_per_epoch 20000 \
      --layer_aware --no_layer_fallback \
      --pseudo_layer_bins 6 --pseudo_layer_knn 18 \
      --neg_layer_margin 1 --layer_gamma 3.0 \
      --neg_hard_ratio $NEGHARD --neg_oversample 10 --neg_adj_ratio $NEGADJ \
      --normal_aware \
      --normal_knn 12 --normal_margin 0.5 --normal_gamma 3.0 \
      --activation prelu --scheduler \
      --lr 5e-4 --weight_decay 1e-4 \
      --seed 42 --device cuda > /dev/null 2>&1

    NPZ_FILE="${PREFIX}.augK2_d64_for_cluster.npz"
    rm -f "${PREFIX}.augK2_d64_none.h5ad"

    if [ ! -f "$NPZ_FILE" ]; then
        echo "❌ [${i}/50] 训练异常，未找到 NPZ 文件，跳过此轮。"
        continue
    fi

    echo "   -> 训练完成，开始聚类 (model.cluster 会自动添加 cluster_results/ 子目录)..."

    # ---------------------------------------------------------
    # 2. 执行四种聚类配置
    # 关键：--out_prefix 直接使用 OUT_DIR 下的前缀（不加 cluster_results）
    #       因为 model.cluster 内部会自动创建 cluster_results/ 目录并将文件放入其中。
    #       最终实际路径为：${OUT_DIR}/cluster_results/trial_${i}_${CFG}.${METHOD}.metrics.txt
    # ---------------------------------------------------------

    # 聚类 A: mclust 极简锐利版
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method mclust --use_rep emb --pca_dim 15 --n_clusters 5 \
      --refine --refine_iter 1 --calc_acc \
      --out_prefix "${OUT_DIR}/trial_${i}_mclust_A" > /dev/null 2>&1

    # 聚类 B: mclust 弧线平滑版
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method mclust --use_rep emb --pca_dim 20 --n_clusters 5 \
      --power 1 --power_alpha 0.5 \
      --refine --refine_iter 1 --calc_acc \
      --out_prefix "${OUT_DIR}/trial_${i}_mclust_B" > /dev/null 2>&1

    # 聚类 C: robust 偏重特征
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method robust --use_rep emb --n_clusters 5 \
      --w_spa 0.3 --w_emb 0.7 --knn_k 10 \
      --refine_k 6 --refine_iter 1 --calc_acc \
      --out_prefix "${OUT_DIR}/trial_${i}_robust_C" > /dev/null 2>&1

    # 聚类 D: robust 均衡拓扑
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method robust --use_rep emb --n_clusters 5 \
      --w_spa 0.6 --w_emb 0.4 --knn_k 12 \
      --refine_k 6 --refine_iter 1 --calc_acc \
      --out_prefix "${OUT_DIR}/trial_${i}_robust_D" > /dev/null 2>&1

    # ---------------------------------------------------------
    # 3. 提取结果并汇总 (从 model.cluster 自动生成的 cluster_results/ 下读取)
    # ---------------------------------------------------------
    for CFG in "mclust_A" "mclust_B" "robust_C" "robust_D"; do
        METHOD="robust"
        [[ "$CFG" == *"mclust"* ]] && METHOD="mclust"

        # 实际文件路径：${OUT_DIR}/cluster_results/trial_${i}_${CFG}.${METHOD}.metrics.txt
        METRICS_FILE="${OUT_DIR}/cluster_results/trial_${i}_${CFG}.${METHOD}.metrics.txt"
        if [ -f "$METRICS_FILE" ]; then
            ARI=$(grep "^ARI" "$METRICS_FILE" | awk '{print $2}')
            NMI=$(grep "^NMI" "$METRICS_FILE" | awk '{print $2}')
            ACC=$(grep "^ACC" "$METRICS_FILE" | awk '{print $2}')
            echo "${i},${ALPHA},${K},${LRECON},${NEGHARD},${NEGADJ},${METHOD},${CFG},${ARI},${NMI},${ACC}" >> "$LOG_FILE"
        else
            echo "⚠️ 警告: 未找到 ${METRICS_FILE}，可能聚类失败，跳过该配置。"
        fi
    done
done

echo "=================================================="
echo "🏆 50 次训练与聚类搜索全部完成！"
echo "Top 10 ARI 参数组合如下："
echo "Trial,Alpha,K_Graph,LambdaRecon,NegHard,NegAdj,Method,Config,ARI,NMI,ACC"
tail -n +2 "$LOG_FILE" | sort -t, -k9 -nr | head -n 10
echo "=================================================="
echo "完整日志已保存至: $LOG_FILE"
echo "聚类结果文件位于: ${OUT_DIR}/cluster_results/"