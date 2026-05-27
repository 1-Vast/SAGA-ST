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

# 全局路径变量 (请确保路径与你环境一致)
H5_INPUT="/root/autodl-tmp/DLPFC/151671_dropNaN.h5ad"
OUT_DIR="/root/autodl-tmp/DLPFC/search_50_trials"
LOG_FILE="${OUT_DIR}/master_summary.csv"

# 创建输出目录和日志表头
mkdir -p "$OUT_DIR"
echo "Trial,Alpha,K_Graph,LambdaRecon,NegHard,NegAdj,Method,Config,ARI,NMI,ACC" > "$LOG_FILE"

# ==========================================
# 🎯 核心超参数池 (深度优化版)
# ==========================================
ALPHAS=(0.2 0.3 0.35 0.4)        # 调低上限：防止过度平滑，保留基因表达的原始差异
KS=(4 6 8)                       # 缩小范围：Visium 是六边形网格，物理一阶邻居就是 6，太大跨层
LRECONS=(0.0 0.01 0.05)          # 暴降权重：切断自编码器重建的干扰，逼迫模型专注边界划分
NEGHARDS=(0.7 0.8 0.85)          # 大幅上调：拉高困难负样本挖掘比例，死磕模糊边界
NEGADJS=(0.3 0.4 0.5)            # 新增维度：强制模型在物理相邻但结构不同的层之间采样负样本

echo "🚀 开始执行 50 次超参数自动化搜索 (专注锐利边界挖掘)..."
echo "=================================================="

for i in {1..50}; do
    # 随机抽取本轮的参数组合
    ALPHA=${ALPHAS[$RANDOM % ${#ALPHAS[@]}]}
    K=${KS[$RANDOM % ${#KS[@]}]}
    LRECON=${LRECONS[$RANDOM % ${#LRECONS[@]}]}
    NEGHARD=${NEGHARDS[$RANDOM % ${#NEGHARDS[@]}]}
    NEGADJ=${NEGADJS[$RANDOM % ${#NEGADJS[@]}]}

    PREFIX="${OUT_DIR}/trial_${i}"
    
    echo "⏳ [${i}/50] 训练特征 | Alpha=${ALPHA}, KNN=${K}, LRecon=${LRECON}, Hard=${NEGHARD}, Adj=${NEGADJ}"

    # 1. 执行 main.py 训练
    # 🚨 关键修改: --embed_agg 改为 mean，严格将输出控制在 64 维，消灭维度灾难！
    # 🚨 关键修改: 提高 normal_gamma 至 3.0，强制施加皮层弧线几何约束！
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

    # 严格控制只保留 NPZ
    NPZ_FILE="${PREFIX}.augK2_d64_for_cluster.npz"
    rm -f "${PREFIX}.augK2_d64_none.h5ad"

    if [ ! -f "$NPZ_FILE" ]; then
        echo "❌ [${i}/50] 训练异常，未找到 NPZ 文件，跳过此轮。"
        continue
    fi

    echo "   -> 训练完成，开始对该 NPZ (64维特征) 执行 4 组精细化聚类..."

    # ---------------------------------------------------------
    # 2. 针对 64 维特征优化后的 4 组聚类配置
    # ---------------------------------------------------------

    # 聚类 A: mclust 极简锐利版 (pca=15, 仅使用 1 次多数投票去噪)
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method mclust --use_rep emb --pca_dim 15 --n_clusters 5 \
      --refine --refine_iter 1 --calc_acc --out_prefix "${PREFIX}_mclust_A" > /dev/null 2>&1

    # 聚类 B: mclust 弧线平滑版 (引入轻量级特征图一跳传播 power=1, 保护结构连续性)
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method mclust --use_rep emb --pca_dim 20 --n_clusters 5 \
      --power 1 --power_alpha 0.5 \
      --refine --refine_iter 1 --calc_acc --out_prefix "${PREFIX}_mclust_B" > /dev/null 2>&1

    # 聚类 C: robust 偏重特征 (w_emb=0.7, 预防大面积同化吞噬薄层)
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method robust --use_rep emb --n_clusters 5 \
      --w_spa 0.3 --w_emb 0.7 --knn_k 10 \
      --refine_k 6 --refine_iter 1 --calc_acc --out_prefix "${PREFIX}_robust_C" > /dev/null 2>&1

    # 聚类 D: robust 均衡拓扑 (w_spa=0.6, 中等空间约束)
    python -m model.cluster --npz "$NPZ_FILE" --h5 "$H5_INPUT" --label_key "sce.layer_guess" \
      --method robust --use_rep emb --n_clusters 5 \
      --w_spa 0.6 --w_emb 0.4 --knn_k 12 \
      --refine_k 6 --refine_iter 1 --calc_acc --out_prefix "${PREFIX}_robust_D" > /dev/null 2>&1

    # ---------------------------------------------------------
    # 3. 提取结果并汇总
    # ---------------------------------------------------------
    for CFG in "mclust_A" "mclust_B" "robust_C" "robust_D"; do
        METHOD="robust"
        [[ "$CFG" == *"mclust"* ]] && METHOD="mclust"

        METRICS_FILE="${PREFIX}_${CFG}.${METHOD}.metrics.txt"
        if [ -f "$METRICS_FILE" ]; then
            ARI=$(grep "^ARI" "$METRICS_FILE" | awk '{print $2}')
            NMI=$(grep "^NMI" "$METRICS_FILE" | awk '{print $2}')
            ACC=$(grep "^ACC" "$METRICS_FILE" | awk '{print $2}')
            # 写入 CSV
            echo "${i},${ALPHA},${K},${LRECON},${NEGHARD},${NEGADJ},${METHOD},${CFG},${ARI},${NMI},${ACC}" >> "$LOG_FILE"
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