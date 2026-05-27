# Demo and Paper Artifacts

`demo/` 目录用于保存论文复现实验的命令记录、绘图脚本和结果组织方式。大体量中间结果、图片、表格和 PDF 默认不进入 Git，请根据 `REPRODUCIBILITY.md` 重新生成。

## Suggested Usage

1. 先按照 `dataset/README.md` 准备数据。
2. 参考对应数据集目录下的 `train/` 命令记录运行 `model.main`。
3. 使用 `model.cluster` 生成聚类标签、指标和空间图。
4. 运行 `plot/` 或 `demo/paper/绘图/` 中的脚本复现论文图。

## Existing Experiment Groups

```text
demo/
+-- DLPFC/                         # DLPFC 切片实验
+-- Brain/                         # 小鼠脑矢状切片实验
+-- coronal mouse brain/            # 小鼠脑冠状切片实验
+-- Human breast cancer/            # 人乳腺癌实验
+-- mouse/                          # 小鼠胚胎实验
+-- paper/实验/                     # 论文实验结果组织
+-- paper/绘图/                     # 论文绘图脚本和图表组织
```

## Output Policy

以下文件通常由实验生成，不建议直接提交到普通 Git：

- `*.h5ad`
- `*.npz`
- `*.npy`
- `*.png`
- `*.pdf`
- `*.xlsx`
- `cluster_results/`

如需公开完整复现包，请使用 Git LFS 或外部归档平台，并在 `README.md` 或 `dataset/README.md` 中补充下载方式。
