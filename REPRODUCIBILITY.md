# Reproducibility Guide

本文档说明如何从代码、环境和外部数据复现论文实验结果。仓库默认只版本控制代码、配置和复现说明；大体量数据、模型中间结果和论文图像产物不直接提交到 Git。

## 1. 环境

```bash
conda env create -f environment.yml
conda activate sp
```

安装与 PyTorch 2.5.0 + CUDA 12.1 匹配的 `torch-geometric`：

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install torch-geometric
```

为了提高复现稳定性，训练前建议设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
```

Windows PowerShell 可使用：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
$env:NUMEXPR_NUM_THREADS="2"
```

## 2. 数据放置

请按照 `dataset/README.md` 中的目录结构准备 `.h5ad` 或 10x Visium 原始文件。训练入口最少需要：

- `adata.X`
- `adata.obsm["spatial"]`
- 可选的 `adata.obs` 标签列，用于计算 ARI、NMI、ACC 等指标

## 3. 快速复现流程

下面以 DLPFC 151507 为例。正式复现实验时，请根据论文和 `demo/` 记录替换对应数据集、聚类数和超参数。

### 3.1 训练 embedding

```bash
python -m model.main \
  --h5 "dataset/DLPFC/151507.h5ad" \
  --out_prefix "outputs/DLPFC/151507/SAGA_ST_151507" \
  --use_scanpy_workflow \
  --graph_model KNN --k 10 \
  --pca_comps 64 \
  --dim 64 --K 2 --hidden 512 --embed_agg concat \
  --alpha 0.55 \
  --topN 50 \
  --lambda_recon 0.10 \
  --mask_ratio_feat 0.30 \
  --epochs 400 \
  --pos_per_epoch 20000 \
  --activation prelu --scheduler \
  --lr 0.001 --weight_decay 1e-4 \
  --seed 42 --device cuda
```

主要输出：

```text
outputs/DLPFC/151507/SAGA_ST_151507.augK2_d64_for_cluster.npz
outputs/DLPFC/151507/SAGA_ST_151507.augK2_d64_none.h5ad
```

### 3.2 聚类和评估

```bash
python -m model.cluster \
  --npz "outputs/DLPFC/151507/SAGA_ST_151507.augK2_d64_for_cluster.npz" \
  --h5 "dataset/DLPFC/151507.h5ad" \
  --label_key "Ground Truth" \
  --method mclust \
  --use_rep emb \
  --n_clusters 7 \
  --pca_dim 32 \
  --refine_k 6 \
  --refine_iter 1 \
  --calc_acc \
  --out_prefix "outputs/DLPFC/151507/SAGA_ST_151507_cluster" \
  --progress
```

聚类结果会写入 `outputs/DLPFC/151507/cluster_results/`，包括：

- `*.metrics.txt`
- `*.labels.txt`
- `*.spatial.png`
- `*.UMAP_2d.png`

## 4. 论文实验入口

已有实验记录和绘图脚本集中在 `demo/`。建议按以下顺序复现：

1. 根据 `dataset/README.md` 准备数据。
2. 参考 `demo/*/*/train/` 中的命令记录训练 embedding。
3. 使用 `model.cluster` 复现聚类标签和指标。
4. 使用 `demo/*/*/plot/` 或 `demo/paper/绘图/` 中的脚本生成论文图。

## 5. 复现检查清单

- Conda 环境创建成功。
- `torch`、`torch_geometric`、`scanpy` 能正常导入。
- 输入 `.h5ad` 包含 `obsm["spatial"]`。
- 训练阶段生成 `.npz` 和 `.h5ad`。
- 聚类阶段生成 `metrics.txt`、`labels.txt` 和空间图。
- 论文表格中的指标使用相同标签列、聚类数和随机种子计算。

## 6. 注意事项

- `mclust` 方法优先调用 R 包 `mclust`；不可用时会回退到 `sklearn GaussianMixture`，数值结果可能略有差异。
- 大文件和生成结果默认被 `.gitignore` 排除。如需归档完整数据，请使用机构存储、Zenodo、Figshare、OSF 或 Git LFS，并在 `dataset/README.md` 中记录下载地址和校验信息。
