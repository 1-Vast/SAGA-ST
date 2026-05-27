# SAGA-ST

本仓库是论文方法 **SAGA-ST** 的代码与复现实验材料，主要用于空间转录组数据的表示学习、空间域聚类、指标评估和论文图表复现。

完整复现流程见 `REPRODUCIBILITY.md`，数据放置规范见 `dataset/README.md`，demo 和论文图表产物说明见 `demo/README.md`。

代码分为两个主要阶段：

1. 使用 `model.main` 在空间转录组数据上训练图表示，输出 spot embedding。
2. 使用 `model.cluster` 基于 embedding 进行聚类、空间平滑、指标评估和结果可视化。

## 目录结构

```text
SAGA-ST/
+-- environment.yml          # Conda 环境配置
+-- model/                   # 核心模型、预处理、训练和聚类代码
|   +-- main.py              # 表示学习入口，输出 embedding
|   +-- cluster.py           # 聚类、评估和可视化入口
|   +-- preprocess.py        # 空间图构建和 Scanpy 预处理
|   +-- encoder.py           # 编码器和损失相关模块
|   +-- augment.py           # 图增强
|   +-- batch.py             # 批处理辅助函数
+-- dataset/                 # 论文实验使用的数据
+-- demo/                    # 复现实验脚本、日志、结果和绘图文件
+-- viz/                     # 可视化辅助代码
```

## 环境配置

建议使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate sp
```

`environment.yml` 中的 PyTorch 版本为 `2.5.0`，CUDA 版本为 `12.1`。`torch-geometric` 及其编译依赖需要根据 PyTorch/CUDA 版本单独安装：

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install torch-geometric
```

如果使用 `mclust` 聚类并希望调用 R 的 `Mclust`，还需要在 R 环境中安装 `mclust`。如果 `rpy2` 或 R `mclust` 不可用，代码会回退到 `sklearn GaussianMixture`。

## 数据准备

训练入口接受 `.h5ad` 格式数据：

```bash
python -m model.main --h5 path/to/data.h5ad
```

输入数据建议包含：

- `adata.X`：基因表达矩阵。
- `adata.obsm["spatial"]`：spot 空间坐标。
- 可选的真实标签列，例如 `adata.obs["Ground Truth"]`、`adata.obs["spatial_domain"]`、`adata.obs["allen_cluster"]`，用于评估 ARI、NMI、ACC 等指标。

仓库中已包含多个论文实验数据集和示例结果，例如：

- `dataset/DLPFC/`
- `dataset/Brain/`
- `dataset/coronal mouse brain/`
- `dataset/Human_Breast_Cancer/`
- `dataset/Mouse embryo/`

## 训练表示

下面示例在 DLPFC 151507 数据上训练 embedding：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

python -m model.main \
  --h5 "dataset/DLPFC/151507.h5ad" \
  --out_prefix "demo/DLPFC/151507/train/151507_reproduce" \
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

训练完成后会生成：

```text
<out_prefix>.augK<K>_d<dim>_for_cluster.npz
<out_prefix>.augK<K>_d<dim>_none.h5ad
```

其中 `.npz` 文件用于后续聚类，`.h5ad` 文件保存了 embedding 到 `adata.obsm`。

## 聚类与评估

使用训练阶段输出的 `.npz` 文件进行聚类：

```bash
python -m model.cluster \
  --npz "demo/DLPFC/151507/train/151507_reproduce.augK2_d64_for_cluster.npz" \
  --h5 "dataset/DLPFC/151507.h5ad" \
  --label_key "Ground Truth" \
  --method mclust \
  --use_rep emb \
  --n_clusters 7 \
  --pca_dim 32 \
  --refine_k 6 \
  --refine_iter 1 \
  --calc_acc \
  --out_prefix "demo/DLPFC/151507/train/151507_reproduce_cluster" \
  --progress
```

聚类结果会保存到 `cluster_results/` 目录，主要输出包括：

```text
*.metrics.txt       # ARI、NMI、ACC、Silhouette 等指标
*.labels.txt        # 每个 spot 的聚类标签
*.spatial.png       # 空间聚类图
*.UMAP_2d.png       # 二维 embedding 可视化
```

`model.cluster` 支持的聚类方法包括：

- `kmeans`
- `leiden`
- `louvain`
- `robust`
- `mclust`

## 论文实验复现

`demo/` 目录中保留了论文实验相关的命令、参数、结果文件和绘图脚本。可优先参考以下路径：

```text
demo/DLPFC/
demo/Brain/
demo/coronal mouse brain/
demo/paper/实验/
demo/paper/绘图/
```

例如，脑切片拼接实验可参考：

```text
demo/Brain/Mouse Brain Serial Section 1/train/brain_S1.txt
demo/Brain/Mouse Brain Serial Section 2/train/brain_S2.txt
```

冠状小鼠脑实验可参考：

```text
demo/coronal mouse brain/train/coronal_mouse_brain.text
```

论文图表和最终实验材料集中在：

```text
demo/paper/实验/
demo/paper/绘图/
```

## 复现建议

为减少随机性，建议固定随机种子并设置确定性 CUDA 变量：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

如果在 CPU 上运行，将命令中的 `--device cuda` 改为：

```bash
--device cpu
```

大规模数据或长 epoch 训练会占用较多显存和时间。快速检查流程时，可以先降低：

- `--epochs`
- `--pos_per_epoch`
- `--hidden`
- `--dim`

正式复现论文结果时，请使用 `demo/` 中对应实验记录的参数。

## 常见问题

### torch-geometric 安装失败

请确认 PyTorch 和 CUDA 版本与安装链接一致。本仓库默认环境为 PyTorch `2.5.0` 和 CUDA `12.1`。

### mclust 不可用

`model.cluster` 会尝试通过 `rpy2` 调用 R `mclust`。如果不可用，会回退到 `sklearn GaussianMixture`，但结果可能与论文中使用的 R `Mclust` 略有差异。

### 找不到空间坐标

请确认输入 `.h5ad` 包含：

```python
adata.obsm["spatial"]
```

如果原始数据来自 10x Visium，可先使用 Scanpy/Squidpy 读取并保存为 `.h5ad`。

## 引用

如果本代码对你的研究有帮助，请引用对应论文。论文正式发表后，可在此处补充 BibTeX 信息。
