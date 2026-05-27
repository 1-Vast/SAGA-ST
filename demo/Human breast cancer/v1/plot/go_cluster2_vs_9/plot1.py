import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import gseapy as gp

# =========================
# 1. 输入文件
# =========================
h5_path = "/root/autodl-tmp/cancer/breast_with_gt.h5ad"
label_path = "/root/autodl-tmp/cancer/cluster_results/breast_v1.augK3_d96_for_cluster.robust.labels.txt"
out_dir = "/root/autodl-tmp/cancer/go_cluster2_vs_9"
os.makedirs(out_dir, exist_ok=True)

target_a = 2   # 关注 cluster 2 相对于 cluster 9
target_b = 9

# 差异基因筛选阈值
padj_cutoff = 0.05
logfc_cutoff = 0.25
top_go_n = 10

# =========================
# 2. 读入数据并合并标签
# =========================
adata = ad.read_h5ad(h5_path)
lab = pd.read_csv(label_path, sep="\t")
lab["cell_id"] = lab["cell_id"].astype(str)

adata.obs["robust_cluster"] = -1
common = adata.obs_names.intersection(lab["cell_id"])
mapper = lab.set_index("cell_id")["cluster"].astype(int)
adata.obs.loc[common, "robust_cluster"] = mapper.loc[common].values

# 只保留 cluster 2 和 9
mask = adata.obs["robust_cluster"].isin([target_a, target_b]).values
adata_sub = adata[mask].copy()

print("subset shape:", adata_sub.shape)
print(adata_sub.obs["robust_cluster"].value_counts().sort_index())

# =========================
# 3. 预处理用于差异分析
#    原始 counts -> normalize_total -> log1p
# =========================
if sp.issparse(adata_sub.X):
    adata_sub.X = adata_sub.X.tocsr()

sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

adata_sub.obs["group"] = adata_sub.obs["robust_cluster"].astype(str)

# =========================
# 4. 差异表达：2 vs 9
# =========================
sc.tl.rank_genes_groups(
    adata_sub,
    groupby="group",
    groups=[str(target_a)],
    reference=str(target_b),
    method="wilcoxon",
    pts=True
)

de_df = sc.get.rank_genes_groups_df(adata_sub, group=str(target_a))
de_df.to_csv(os.path.join(out_dir, "DE_cluster2_vs_9.csv"), index=False)

# 兼容不同 scanpy 输出列名
padj_col = None
for c in ["pvals_adj", "pvals_adj"]:
    if c in de_df.columns:
        padj_col = c
        break

logfc_col = None
for c in ["logfoldchanges", "log2fc", "logFC"]:
    if c in de_df.columns:
        if c in de_df.columns:
            logfc_col = c
            break

if padj_col is None or logfc_col is None:
    raise ValueError(f"DE结果缺少必要列。现有列: {de_df.columns.tolist()}")

# 筛选 cluster 2 上调基因
up_df = de_df[
    (de_df[padj_col] < padj_cutoff) &
    (de_df[logfc_col] > logfc_cutoff)
].copy()

up_df = up_df.dropna(subset=["names"])
up_genes = up_df["names"].astype(str).drop_duplicates().tolist()

print(f"significant up genes in cluster {target_a}: {len(up_genes)}")

if len(up_genes) < 5:
    raise ValueError("显著上调基因太少，GO 富集可能不稳定。请放宽阈值后重试。")

# 背景基因集：本次参与分析的所有基因
background = adata_sub.var_names.astype(str).tolist()

# =========================
# 5. GO 富集分析
#    人乳腺癌数据 -> human
# =========================
enr = gp.enrichr(
    gene_list=up_genes,
    gene_sets="GO_Biological_Process_2023",
    background=background,
    organism="human",
    outdir=None,
)

go_res = enr.results.copy()
go_res.to_csv(os.path.join(out_dir, "GO_cluster2_vs_9_raw.csv"), index=False)

# 统一列名
if "Adjusted P-value" not in go_res.columns:
    raise ValueError(f"GO结果缺少 'Adjusted P-value' 列。现有列: {go_res.columns.tolist()}")

if "Term" not in go_res.columns:
    raise ValueError(f"GO结果缺少 'Term' 列。现有列: {go_res.columns.tolist()}")

go_plot = go_res.copy()
go_plot = go_plot[go_plot["Adjusted P-value"] < 0.05].copy()
go_plot["neglog10_padj"] = -np.log10(go_plot["Adjusted P-value"].clip(lower=1e-300))
go_plot = go_plot.sort_values("neglog10_padj", ascending=False).head(top_go_n).copy()

if go_plot.empty:
    raise ValueError("没有显著 GO term。可以尝试放宽差异基因阈值。")

# 去掉 GO term 后面的编号（如果有）
go_plot["Term_clean"] = go_plot["Term"].astype(str).str.replace(r"\s*\(GO:\d+\)$", "", regex=True)

# 为了画图从上到下递减，倒序
go_plot = go_plot.iloc[::-1].copy()
go_plot.to_csv(os.path.join(out_dir, "GO_cluster2_vs_9_top10_for_plot.csv"), index=False)

# =========================
# 6. 画图
# =========================
plt.figure(figsize=(8.6, 5.6))
bars = plt.barh(
    go_plot["Term_clean"],
    go_plot["neglog10_padj"],
    height=0.62,
    color="#f4a09c",
    edgecolor="white"
)

plt.xlabel(r"$-log_{10}$ (Adjusted P-value)", fontsize=13, fontweight="bold")
plt.ylabel("")
plt.title(f"GO terms enriched in cluster {target_a} vs cluster {target_b}", fontsize=14, pad=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=11)

plt.tight_layout()
png_path = os.path.join(out_dir, "GO_cluster2_vs_9_barplot.png")
pdf_path = os.path.join(out_dir, "GO_cluster2_vs_9_barplot.pdf")
plt.savefig(png_path, dpi=400, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("\nDone.")
print("DE file:", os.path.join(out_dir, "DE_cluster2_vs_9.csv"))
print("GO raw :", os.path.join(out_dir, "GO_cluster2_vs_9_raw.csv"))
print("GO top :", os.path.join(out_dir, "GO_cluster2_vs_9_top10_for_plot.csv"))
print("PNG    :", png_path)
print("PDF    :", pdf_path)