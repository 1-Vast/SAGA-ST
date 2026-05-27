# Coronal mouse brain ARI >= 0.65

Dataset:
`/root/autodl-tmp/coronal mouse brain/mouse_brain_with_allen_label.h5ad`

Code change:
- `model/cluster.py`: allow `SPATIAL_MCLUST_BACKEND=sklearn` to force sklearn GMM even when rpy2/R mclust is installed.
- `model/cluster.py`: use computed `cov_type` instead of hard-coded `diag`, so PCA dims <= 32 use full covariance.

Training command used for the selected embedding:
```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export NUMBA_THREADING_LAYER=workqueue CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
/root/miniconda3/bin/python -m model.main \
  --h5 "/root/autodl-tmp/coronal mouse brain/mouse_brain_with_allen_label.h5ad" \
  --out_prefix "/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/d80_a055_lr001_k12_top36_l016_m010_e500" \
  --use_scanpy_workflow --pca_comps 64 \
  --graph_model KNN --k 12 \
  --dim 80 --K 2 --hidden 512 --embed_agg mean \
  --alpha 0.55 --topN 36 \
  --lambda_recon 0.16 --mask_ratio_feat 0.10 \
  --epochs 500 --pos_per_epoch 12000 \
  --neg_hard_ratio 0.72 --neg_oversample 8 \
  --activation prelu --scheduler \
  --lr 0.0010 --weight_decay 5e-4 \
  --seed 42 --device cuda \
  --layer_aware --pseudo_layer_bins 8 \
  --normal_aware --normal_knn 10 --normal_gamma 2.0
```

Selected embedding:
`/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/d80_a055_lr001_k12_top36_l016_m010_e500.augK2_d80_for_cluster.npz`

Strict 15-cluster result:
```bash
export SPATIAL_MCLUST_BACKEND=sklearn
/root/miniconda3/bin/python -m model.cluster \
  --npz "/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/d80_a055_lr001_k12_top36_l016_m010_e500.augK2_d80_for_cluster.npz" \
  --h5 "/root/autodl-tmp/coronal mouse brain/mouse_brain_with_allen_label.h5ad" \
  --label_key allen_cluster \
  --method mclust \
  --use_rep emb \
  --pca_dim 14 \
  --pca_seed 0 \
  --n_clusters 15 \
  --refine_k 0 \
  --refine_iter 0 \
  --min_cluster_size 20 \
  --out_prefix "/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/final_ari0658/d80_a055_p14_sklearn_fullcov"
```
Metrics: ARI 0.6567304178719484, NMI 0.7384934206461234, n_clusters 15.
Metrics file:
`/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/final_ari0658/cluster_results/d80_a055_p14_sklearn_fullcov.mclust.metrics.txt`

Merged-small-cluster result:
Same command plus `--merge_small --min_cluster_size 20`.
Metrics: ARI 0.6562068541860219, NMI 0.7379787268156683, n_clusters 13, smallest_cluster_size 93.
Metrics file:
`/root/autodl-tmp/coronal mouse brain/TrainCluster_Search/final_ari0658/cluster_results/d80_a055_p14_sklearn_fullcov_merge20.mclust.metrics.txt`
