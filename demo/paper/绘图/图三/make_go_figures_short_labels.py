from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


base = Path(__file__).resolve().parent
source = base / "remote_source"

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 420,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#222222",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def draw_barplot():
    df = pd.read_csv(source / "go_cluster2_vs_9__GO_cluster2_vs_9_top10_for_plot.csv")
    label_map = {
        "Sphingolipid Metabolic Process": "Sphingolipid",
        "Negative Regulation Of Programmed Cell Death": "CellDeath",
        "Response To Estrogen": "Estrogen",
        "Negative Regulation Of DNA-templated Transcription": "Repression",
        "Negative Regulation Of Apoptotic Process": "Anti-apoptosis",
        "acyl-CoA Biosynthetic Process": "Acyl-CoA",
        "Membrane Lipid Biosynthetic Process": "MembraneLipid",
        "Fatty Acid Biosynthetic Process": "FattyAcid",
        "Regulation Of Apoptotic Process": "Apoptosis",
        "Regulation Of Translation": "Translation",
    }
    df["Short"] = df["Term_clean"].map(label_map).fillna(df["Term_clean"])
    df = df.sort_values("neglog10_padj", ascending=False).copy()

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.barh(
        y,
        df["neglog10_padj"].to_numpy(float),
        color="#f49a9a",
        edgecolor="none",
        height=0.58,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["Short"].tolist(), fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(4.7, float(df["neglog10_padj"].max()) * 1.04))
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.tick_params(axis="x", labelsize=8, length=3, width=0.8)
    ax.tick_params(axis="y", length=3, width=0.8, pad=3)
    ax.set_xlabel(r"$-\log_{10}$(adj. P)", fontsize=9, labelpad=5)
    ax.set_title("GO enrichment: C2 vs C9", fontsize=10, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.18, top=0.86)
    fig.savefig(base / "4.png", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base / "4.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def draw_dotplot():
    df = pd.read_csv(source / "go_domain1_and_11_final_v3__spatial_domain_1.GO_top10.csv")
    label_map = {
        "Translation": "Translation",
        "Cytoplasmic Translation": "Cytoplasm",
        "Peptide Biosynthetic Process": "Peptide",
        "Regulation of Transcription by RNA Polymerase II": "PolII",
        "Gene Expression": "Expression",
        "Macromolecule Biosynthetic Process": "Biosynthesis",
        "Regulation of Apoptotic Process": "Apoptosis",
        "Positive Regulation of DNA-templated Transcription": "Activation",
        "Negative Regulation of DNA-templated Transcription": "Repression",
        "Regulation of DNA-templated Transcription": "Transcription",
    }
    df["Short"] = df["Term_clean"].map(label_map).fillna(df["Term_clean"])
    df = df.sort_values("neglog10_fdr", ascending=False).head(10).copy()

    y = np.arange(len(df))
    x = df["HitCount"].to_numpy(float)
    ratio = df["GeneRatio"].to_numpy(float)
    color = df["neglog10_fdr"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(4.65, 3.65))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.9)
    sc = ax.scatter(
        x,
        y,
        s=1120 * ratio + 18,
        c=color,
        cmap="autumn_r",
        norm=Normalize(vmin=18, vmax=24.6),
        edgecolors="none",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["Short"].tolist(), fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 250)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.tick_params(axis="x", labelsize=8, length=3, width=0.8)
    ax.tick_params(axis="y", length=0, pad=4)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(0.95)
        ax.spines[side].set_color("#222222")

    legend_vals = [0.03, 0.08]
    handles = [
        ax.scatter([], [], s=1120 * val + 18, color="#8a8a8a", edgecolors="none")
        for val in legend_vals
    ]
    leg = ax.legend(
        handles,
        [f"{val:.2f}" for val in legend_vals],
        title="Gene ratio",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.04, 0.90),
        borderaxespad=0,
        labelspacing=1.1,
        handletextpad=0.9,
        fontsize=8,
        title_fontsize=8.5,
    )
    ax.add_artist(leg)

    cax = fig.add_axes([0.83, 0.20, 0.035, 0.42])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$-\log_{10}$(FDR)", fontsize=8.5, labelpad=5)
    cb.ax.tick_params(labelsize=8, length=2.5, width=0.8)
    cb.outline.set_linewidth(0.85)

    fig.subplots_adjust(left=0.26, right=0.78, bottom=0.14, top=0.97)
    fig.savefig(base / "3.png", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base / "3.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    draw_barplot()
    draw_dotplot()
