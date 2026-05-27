import os
import math
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gseapy as gp

# =========================================================
# 1. 输入参数
# =========================================================
h5_path = "/root/autodl-tmp/cancer/breast_with_gt.h5ad"
label_path = "/root/autodl-tmp/cancer/cluster_results/breast_v1.augK3_d96_for_cluster.robust.labels.txt"
out_dir = "/root/autodl-tmp/cancer/go_domain1_and_11_final_v3"
os.makedirs(out_dir, exist_ok=True)

cluster_a = 1
cluster_b = 11

padj_cutoff = 0.05
logfc_cutoff = 0.25
top_go_n = 10

fig_dpi = 600
figsize = (7.5, 4.35)   # 比上一版再宽一点
cmap = "autumn_r"
show_title = False

# =========================================================
# 2. 工具函数
# =========================================================
def truncate_label(s, max_len=58):
    s = str(s)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."

def clean_go_term(s):
    s = str(s)
    s = s.replace("_", " ")
    s = s.replace(" Of ", " of ")
    s = s.replace(" And ", " and ")
    s = s.replace(" Via ", " via ")
    s = s.replace(" To ", " to ")
    s = s.replace(" By ", " by ")
    s = s.replace(" With ", " with ")
    s = s.replace(" In ", " in ")
    return s

def infer_nice_xlim(hit_counts):
    hit_counts = np.asarray(hit_counts, dtype=float)
    vmax = float(np.nanmax(hit_counts))

    if vmax <= 5:
        return 6
    elif vmax <= 10:
        return 12
    elif vmax <= 20:
        return 24
    elif vmax <= 40:
        return 45
    elif vmax <= 60:
        return 70
    elif vmax <= 100:
        return 110
    elif vmax <= 150:
        return 170
    elif vmax <= 220:
        return 250
    elif vmax <= 320:
        return 350
    elif vmax <= 450:
        return 500
    else:
        return int(math.ceil(vmax * 1.12 / 50.0) * 50)

def infer_xticks(xmax):
    if xmax <= 12:
        return np.arange(0, xmax + 1e-6, 2)
    elif xmax <= 24:
        return np.arange(0, xmax + 1e-6, 4)
    elif xmax <= 45:
        return np.arange(0, xmax + 1e-6, 10)
    elif xmax <= 70:
        return np.arange(0, xmax + 1e-6, 10)
    elif xmax <= 110:
        return np.arange(0, xmax + 1e-6, 20)
    elif xmax <= 170:
        return np.arange(0, xmax + 1e-6, 50)
    elif xmax <= 250:
        return np.arange(0, xmax + 1e-6, 50)
    elif xmax <= 350:
        return np.arange(0, xmax + 1e-6, 100)
    elif xmax <= 500:
        return np.arange(0, xmax + 1e-6, 100)
    else:
        step = int(round(xmax / 5 / 50.0) * 50)
        step = max(step, 100)
        return np.arange(0, xmax + 1e-6, step)

def get_up_genes(de_df, padj_cutoff=0.05, logfc_cutoff=0.25):
    up_df = de_df[
        (de_df["pvals_adj"] < padj_cutoff) &
        (de_df["logfoldchanges"] > logfc_cutoff)
    ].copy()
    up_df = up_df.dropna(subset=["names"])
    up_genes = up_df["names"].astype(str).drop_duplicates().tolist()
    return up_df, up_genes

def parse_go_hit_info(go_df, query_gene_n):
    go_df = go_df.copy()

    if "Overlap" in go_df.columns:
        ratio = go_df["Overlap"].astype(str).str.split("/", expand=True)
        hit = pd.to_numeric(ratio[0], errors="coerce")
        denom = pd.to_numeric(ratio[1], errors="coerce")
        go_df["HitCount"] = hit.astype(float)
        go_df["GeneRatio"] = hit / denom.replace(0, np.nan)
        return go_df

    if "Genes" in go_df.columns:
        def count_genes(s):
            if pd.isna(s):
                return 0
            s = str(s).strip()
            if s == "":
                return 0
            if ";" in s:
                arr = [x.strip() for x in s.split(";") if x.strip()]
            else:
                arr = [x.strip() for x in s.split(",") if x.strip()]
            return len(arr)

        go_df["HitCount"] = go_df["Genes"].apply(count_genes).astype(float)
        go_df["GeneRatio"] = go_df["HitCount"] / max(float(query_gene_n), 1.0)
        return go_df

    raise ValueError(
        f"GO result has neither 'Overlap' nor 'Genes'. columns={go_df.columns.tolist()}"
    )

def run_go(gene_list, group_name, out_dir, background, top_go_n=10):
    if len(gene_list) < 5:
        print(f"[WARN] {group_name}: too few up genes, skip GO.")
        return None

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets="GO_Biological_Process_2023",
        background=background,
        organism="human",
        outdir=None,
    )

    go_res = enr.results.copy()
    raw_csv = os.path.join(out_dir, f"{group_name}.GO_raw.csv")
    go_res.to_csv(raw_csv, index=False)
    print("saved:", raw_csv)

    if "Adjusted P-value" not in go_res.columns:
        print(f"[WARN] {group_name}: no 'Adjusted P-value' column.")
        return None

    go_plot = go_res[go_res["Adjusted P-value"] < 0.05].copy()
    if go_plot.empty:
        print(f"[WARN] {group_name}: no significant GO terms.")
        return None

    go_plot["Term_clean"] = go_plot["Term"].astype(str).str.replace(
        r"\s*\(GO:\d+\)$", "", regex=True
    )
    go_plot["Term_clean"] = go_plot["Term_clean"].map(clean_go_term)
    go_plot["neglog10_fdr"] = -np.log10(
        pd.to_numeric(go_plot["Adjusted P-value"], errors="coerce").clip(lower=1e-300)
    )

    go_plot = parse_go_hit_info(go_plot, len(gene_list))
    go_plot = go_plot.dropna(subset=["HitCount", "GeneRatio", "neglog10_fdr"]).copy()

    if go_plot.empty:
        print(f"[WARN] {group_name}: no valid GO terms after parsing.")
        return None

    go_plot = go_plot.sort_values(
        ["neglog10_fdr", "HitCount"],
        ascending=[False, False]
    ).head(top_go_n).copy()

    top_csv = os.path.join(out_dir, f"{group_name}.GO_top{top_go_n}.csv")
    go_plot.to_csv(top_csv, index=False)
    print("saved:", top_csv)

    return go_plot

def plot_single_go(go_df, out_png, out_pdf, domain_id):
    if go_df is None or go_df.empty:
        print(f"[WARN] Spatial domain {domain_id}: no data to plot.")
        return

    df = go_df.copy()
    df = df.sort_values("neglog10_fdr", ascending=False).head(top_go_n).copy()
    df["Term_show"] = df["Term_clean"].map(lambda z: truncate_label(z, 58))
    df = df.iloc[::-1].copy()

    y = np.arange(len(df))
    x = df["HitCount"].astype(float).values
    c = df["neglog10_fdr"].astype(float).values
    ratio_vals = df["GeneRatio"].astype(float).values

    qlow = np.percentile(ratio_vals, 10)
    qhigh = np.percentile(ratio_vals, 90)
    ratio_clip = np.clip(ratio_vals, qlow, qhigh if qhigh > qlow else ratio_vals.max())
    s = 720 * (ratio_clip / max(ratio_clip.max(), 1e-8)) + 35

    xmax = infer_nice_xlim(x)
    xticks = infer_xticks(xmax)

    fig, ax = plt.subplots(figsize=figsize)

    sca = ax.scatter(
        x, y,
        s=s,
        c=c,
        cmap=cmap,
        edgecolors="none"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["Term_show"].tolist(), fontsize=9.5)
    ax.set_xlim(0, xmax)

    if xticks is not None:
        ax.set_xticks(xticks)

    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", length=0)

    ax.grid(axis="y", color="#d0d0d0", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)

    if show_title:
        ax.set_title(f"Spatial domain {domain_id}", fontsize=11, pad=8)

    # 让主图更靠左，右侧腾出更舒展空间
    plt.subplots_adjust(left=0.39, right=0.76, top=0.95, bottom=0.12)

    # -------- size legend --------
    v1 = np.quantile(ratio_vals, 0.35)
    v2 = np.quantile(ratio_vals, 0.75)
    size_vals = np.unique(np.round([v1, v2], 2))

    handles = []
    labels = []
    for v in size_vals:
        vv = np.clip(v, qlow, qhigh if qhigh > qlow else ratio_vals.max())
        ss = 720 * (vv / max(ratio_clip.max(), 1e-8)) + 35
        handles.append(plt.scatter([], [], s=ss, color="gray", edgecolors="none"))
        labels.append(f"{v:.2f}")

    leg = ax.legend(
        handles,
        labels,
        title="% Genes\nin set",
        loc="upper left",
        bbox_to_anchor=(1.03, 0.93),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=1.0,
        handletextpad=0.9,
    )
    plt.setp(leg.get_title(), fontsize=9)
    for txt in leg.get_texts():
        txt.set_fontsize(8.8)

    # -------- colorbar 单独更靠右、更靠下 --------
    cax = fig.add_axes([0.800, 0.31, 0.022, 0.34])
    cbar = fig.colorbar(sca, cax=cax)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label(r"$\log_{10}\frac{1}{FDR}$", fontsize=9, rotation=90, labelpad=12)

    plt.savefig(out_png, dpi=fig_dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("saved:", out_png)
    print("saved:", out_pdf)

# =========================================================
# 3. 读数据
# =========================================================
print("Loading h5ad...")
adata = ad.read_h5ad(h5_path)

print("Loading labels...")
lab = pd.read_csv(label_path, sep="\t")
lab["cell_id"] = lab["cell_id"].astype(str)

adata.obs["robust_cluster"] = -1
common = adata.obs_names.intersection(lab["cell_id"])
mapper = lab.set_index("cell_id")["cluster"].astype(int)
adata.obs.loc[common, "robust_cluster"] = mapper.loc[common].values

mask = adata.obs["robust_cluster"].isin([cluster_a, cluster_b]).values
adata_sub = adata[mask].copy()

if sp.issparse(adata_sub.X):
    adata_sub.X = adata_sub.X.tocsr()

# =========================================================
# 4. 预处理 + DE
# =========================================================
print("Normalizing and log1p...")
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

adata_sub.obs["group"] = adata_sub.obs["robust_cluster"].astype(str)
background = adata_sub.var_names.astype(str).tolist()

print("Running differential expression...")
sc.tl.rank_genes_groups(
    adata_sub,
    groupby="group",
    groups=[str(cluster_a), str(cluster_b)],
    reference="rest",
    method="wilcoxon",
    pts=True,
)

de_a = sc.get.rank_genes_groups_df(adata_sub, group=str(cluster_a)).copy()
de_b = sc.get.rank_genes_groups_df(adata_sub, group=str(cluster_b)).copy()

de_a.to_csv(os.path.join(out_dir, f"DE_spatial_domain_{cluster_a}.csv"), index=False)
de_b.to_csv(os.path.join(out_dir, f"DE_spatial_domain_{cluster_b}.csv"), index=False)

up_a_df, up_a = get_up_genes(de_a, padj_cutoff, logfc_cutoff)
up_b_df, up_b = get_up_genes(de_b, padj_cutoff, logfc_cutoff)

# =========================================================
# 5. GO 富集
# =========================================================
go_a = run_go(up_a, f"spatial_domain_{cluster_a}", out_dir, background, top_go_n)
go_b = run_go(up_b, f"spatial_domain_{cluster_b}", out_dir, background, top_go_n)

# =========================================================
# 6. 作图
# =========================================================
png_a = os.path.join(out_dir, f"Spatial_domain_{cluster_a}_GO_terms.png")
pdf_a = os.path.join(out_dir, f"Spatial_domain_{cluster_a}_GO_terms.pdf")

png_b = os.path.join(out_dir, f"Spatial_domain_{cluster_b}_GO_terms.png")
pdf_b = os.path.join(out_dir, f"Spatial_domain_{cluster_b}_GO_terms.pdf")

plot_single_go(go_a, png_a, pdf_a, domain_id=cluster_a)
plot_single_go(go_b, png_b, pdf_b, domain_id=cluster_b)

print("\nDone.")
print("Output dir:", out_dir)