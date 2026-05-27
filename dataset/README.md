# Dataset Layout

本目录用于放置论文复现实验所需数据。由于 `.h5ad`、`.npz`、10x 原始矩阵和图像文件通常较大，这些文件默认不提交到 Git。请从论文数据来源、公开数据库或作者提供的归档中下载后，按下面结构放置。

## Expected Structure

```text
dataset/
+-- DLPFC/
|   +-- 151507.h5ad
|   +-- 151508.h5ad
|   +-- ...
|   +-- 151676.h5ad
+-- Brain/
|   +-- Mouse Brain Serial Section 1/
|   |   +-- Sagittal-Anterior/
|   |   +-- Sagittal-Posterior/
|   +-- Mouse Brain Serial Section 2/
|       +-- Sagittal-Anterior/
|       +-- Sagittal-Posterior/
+-- coronal mouse brain/
|   +-- mouse_brain_with_allen_label.h5ad
+-- Human_Breast_Cancer/
|   +-- breast_with_gt.h5ad
|   +-- metadata.tsv
|   +-- V1_Human_Breast_Cancer_Block_A_Section_1/
+-- Mouse embryo/
    +-- E11.5_E1S1.MOSTA.h5ad
    +-- E12.5_E1S2.MOSTA.h5ad
    +-- ...
```

## Required AnnData Fields

`model.main` 训练入口要求输入 `.h5ad` 至少包含：

- `adata.X`：表达矩阵。
- `adata.obsm["spatial"]`：空间坐标。

如果需要评估聚类指标，`model.cluster` 还需要指定标签列，例如：

- `Ground Truth`
- `spatial_domain`
- `allen_cluster`

使用示例：

```bash
python -m model.cluster \
  --npz "outputs/example.augK2_d64_for_cluster.npz" \
  --h5 "dataset/DLPFC/151507.h5ad" \
  --label_key "Ground Truth" \
  --method mclust \
  --n_clusters 7
```

## Versioning Policy

- 原始数据和中间结果保存在本地或外部数据仓库。
- Git 仓库只保存代码、环境、复现说明和必要的小型文本配置。
- 如果后续发布数据下载链接，请在本文件补充 URL、文件大小、版本日期和校验和。
