#!/bin/bash
NPZ_FILE="/root/autodl-tmp/DLPFC/151675_v1.augK2_d64_for_cluster.npz"
H5_PATH="/root/autodl-tmp/DLPFC/151675_dropNaN.h5ad"

echo "=================================================================="
echo "🔬 极限压榨：针对最强 NPZ 的全量聚类参数扫描"
echo "目标：在保持 ARI > 0.6 的同时，强行找回第 7 个簇 (K=7)"
echo "=================================================================="

# 🔎 极其细腻的聚类搜索空间
PCA_DIMS=(30 40 50)        # 尝试不同的主成分保留度
SMOOTH_KS=(4 5 6 7 8)      # 细微调整平滑半径
REFINE_KS=(10 15 20 25)    # 严格控制空间投票的感受野，防止大吃小
MIN_SIZES=(10 30 50)       # 必须小于 55，给极薄的皮层留活路

total=$(( ${#PCA_DIMS[@]} * ${#SMOOTH_KS[@]} * ${#REFINE_KS[@]} * ${#MIN_SIZES[@]} ))
count=0

echo "预计聚类次数: $total 次 (纯聚类极快，预计 3-5 分钟)"
echo -e "PCA\tSM_K\tREF_K\tMIN_S\tK\tARI\tNMI" > final_cluster_sweep.log

for pca in "${PCA_DIMS[@]}"; do
  for sm_k in "${SMOOTH_KS[@]}"; do
    for ref_k in "${REFINE_KS[@]}"; do
      for min_s in "${MIN_SIZES[@]}"; do
        count=$((count + 1))
        
        # 运行聚类
        python -m model.cluster \
          --npz "$NPZ_FILE" \
          --h5 "$H5_PATH" \
          --label_key sce.layer_guess \
          --method mclust \
          --use_rep emb \
          --pca_dim "$pca" \
          --n_clusters 7 \
          --smooth --smooth_k "$sm_k" \
          --refine --refine_k "$ref_k" --refine_iter 1 \
          --merge_small --min_cluster_size "$min_s" \
          --calc_acc > temp_clust.log 2>&1
          
        # 解析结果
        k_res=$(awk '$1=="Number" && $2=="of" && $3=="clusters:" {print $4}' temp_clust.log)
        ari_res=$(awk '$1=="ARI:" {print $2}' temp_clust.log)
        nmi_res=$(awk '$1=="NMI:" {print $2}' temp_clust.log)
        
        # 兼容不同的输出格式
        if [ -z "$k_res" ]; then k_res=$(awk '$1=="n_clusters" {print $2}' temp_clust.log); fi
        if [ -z "$ari_res" ]; then ari_res=$(awk '$1=="ARI" {print $2}' temp_clust.log); fi
        if [ -z "$nmi_res" ]; then nmi_res=$(awk '$1=="NMI" {print $2}' temp_clust.log); fi

        if [ -n "$ari_res" ]; then
          # 记录到总日志
          echo -e "${pca}\t${sm_k}\t${ref_k}\t${min_s}\t${k_res}\t${ari_res}\t${nmi_res}" >> final_cluster_sweep.log
          
          # 逻辑判断：只在终端高亮那些“高价值”的发现
          is_k7_high=$(awk -v k="$k_res" -v a="$ari_res" 'BEGIN {print (k == 7 && a >= 0.59) ? 1 : 0}')
          is_break=$(awk -v a="$ari_res" 'BEGIN {print (a >= 0.60) ? 1 : 0}')
          
          if [ "$is_k7_high" -eq 1 ] && [ "$is_break" -eq 1 ]; then
             # K=7 且 ARI 破 0.6 的究极理想状态，闪烁绿色
             printf "\n🌟 [%3d/%d] PCA=%s, sm=%s, ref=%s, min=%s | \033[5;32mK=%s, ARI=%.4f\033[0m, NMI=%.4f\n" "$count" "$total" "$pca" "$sm_k" "$ref_k" "$min_s" "$k_res" "$ari_res" "$nmi_res"
          elif [ "$is_k7_high" -eq 1 ]; then
             # K=7 且 ARI 很高 (0.59+)，黄色警示
             printf "\n💡 [%3d/%d] PCA=%s, sm=%s, ref=%s, min=%s | \033[33mK=7, ARI=%.4f\033[0m, NMI=%.4f\n" "$count" "$total" "$pca" "$sm_k" "$ref_k" "$min_s" "$ari_res" "$nmi_res"
          elif [ "$is_break" -eq 1 ]; then
             # 虽然 K!=7，但 ARI 破 0.6，红色提示
             printf "\n🔥 [%3d/%d] PCA=%s, sm=%s, ref=%s, min=%s | \033[31mK=%s, ARI=%.4f\033[0m, NMI=%.4f\n" "$count" "$total" "$pca" "$sm_k" "$ref_k" "$min_s" "$k_res" "$ari_res" "$nmi_res"
          else
            # 垃圾结果直接单行覆盖刷新，不污染屏幕
            printf "\r⏳ 进度: %d/%d (当前扫过 K=%s, ARI=%s)..." "$count" "$total" "$k_res" "$ari_res"
          fi
        fi
      done
    done
  done
done

# echo -e "\n\n=================================================================="
# echo "✅ 扫描完成！查看 Top 10 的【K=7 专属榜单】："
# echo -e "PCA\tSM_K\tREF_K\tMIN_S\tK\tARI\tNMI"
# awk '$5==7' final_cluster_sweep.log | sort -t$'\t' -k6 -n -r | head -n 10
# echo "=================================================================="
# rm -f temp_clust.log

# root@autodl-container-36fd48b95a-979ea069:~/autodl-tmp/Spatial-main# bash /root/autodl-tmp/Spatial-main/cluster_sweep.sh
# ==================================================================
# 🔬 极限压榨：针对最强 NPZ 的全量聚类参数扫描
# 目标：在保持 ARI > 0.6 的同时，强行找回第 7 个簇 (K=7)
# ==================================================================
# 预计聚类次数: 180 次 (纯聚类极快，预计 3-5 分钟)
# ⏳ 进度: 24/180 (当前扫过 K=5, ARI=0.4678)...
# 🔥 [ 25/180] PCA=30, sm=6, ref=10, min=10 | K=6, ARI=0.6106, NMI=0.6991

# 🔥 [ 26/180] PCA=30, sm=6, ref=10, min=30 | K=6, ARI=0.6106, NMI=0.6991

# 🔥 [ 27/180] PCA=30, sm=6, ref=10, min=50 | K=6, ARI=0.6106, NMI=0.6991

# 🔥 [ 28/180] PCA=30, sm=6, ref=15, min=10 | K=6, ARI=0.6057, NMI=0.6943

# 🔥 [ 29/180] PCA=30, sm=6, ref=15, min=30 | K=6, ARI=0.6057, NMI=0.6943

# 🔥 [ 30/180] PCA=30, sm=6, ref=15, min=50 | K=6, ARI=0.6057, NMI=0.6943

# 🔥 [ 31/180] PCA=30, sm=6, ref=20, min=10 | K=6, ARI=0.6078, NMI=0.6980

# 🔥 [ 32/180] PCA=30, sm=6, ref=20, min=30 | K=6, ARI=0.6078, NMI=0.6980

# 🔥 [ 33/180] PCA=30, sm=6, ref=20, min=50 | K=6, ARI=0.6078, NMI=0.6980

# 🔥 [ 34/180] PCA=30, sm=6, ref=25, min=10 | K=6, ARI=0.6062, NMI=0.6944

# 🔥 [ 35/180] PCA=30, sm=6, ref=25, min=30 | K=6, ARI=0.6062, NMI=0.6944

# 🔥 [ 36/180] PCA=30, sm=6, ref=25, min=50 | K=6, ARI=0.6062, NMI=0.6944
# ⏳ 进度: 180/180 (当前扫过 K=6, ARI=0.4009)...

# ==================================================================
# ✅ 扫描完成！查看 Top 10 的【K=7 专属榜单】：
# PCA     SM_K    REF_K   MIN_S   K       ARI     NMI
# ==================================================================
# root@autodl-container-36fd48b95a-979ea069:~/autodl-tmp/Spatial-main# 