from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(r"D:\Spatial-main")
DATA_ROOT = ROOT / "dataset" / "Brain"
OUT_DIR = ROOT / "demo" / "\u5b9e\u9a8c" / "experiment4" / "gene_highlight_single"

SECTION_1 = {
    "anterior_h5": DATA_ROOT
    / "Mouse Brain Serial Section 1"
    / "Sagittal-Anterior"
    / "V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5",
    "anterior_pos": DATA_ROOT
    / "Mouse Brain Serial Section 1"
    / "Sagittal-Anterior"
    / "spatial"
    / "tissue_positions_list.csv",
    "posterior_h5": DATA_ROOT
    / "Mouse Brain Serial Section 1"
    / "Sagittal-Posterior"
    / "V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5",
    "posterior_pos": DATA_ROOT
    / "Mouse Brain Serial Section 1"
    / "Sagittal-Posterior"
    / "spatial"
    / "tissue_positions_list.csv",
}

SECTION_2 = {
    "anterior_h5": DATA_ROOT
    / "Mouse Brain Serial Section 2"
    / "Sagittal-Anterior"
    / "V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix.h5",
    "anterior_pos": DATA_ROOT
    / "Mouse Brain Serial Section 2"
    / "Sagittal-Anterior"
    / "spatial"
    / "tissue_positions.csv",
    "posterior_h5": DATA_ROOT
    / "Mouse Brain Serial Section 2"
    / "Sagittal-Posterior"
    / "V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix.h5",
    "posterior_pos": DATA_ROOT
    / "Mouse Brain Serial Section 2"
    / "Sagittal-Posterior"
    / "spatial"
    / "tissue_positions.csv",
}

PANELS = [
    {
        "title": "Gpsm1",
        "gene": "Gpsm1",
        "section": "section1",
        "piece": "anterior",
        "filename": "Gpsm1.png",
    },
    {
        "title": "Gpr88",
        "gene": "Gpr88",
        "section": "section1",
        "piece": "anterior",
        "filename": "Gpr88.png",
    },
    {
        "title": "Cbln3",
        "gene": "Cbln3",
        "section": "section1",
        "piece": "posterior",
        "filename": "Cbln3.png",
    },
    {
        "title": "Hpca(section 1)",
        "gene": "Hpca",
        "section": "section1",
        "piece": "posterior",
        "filename": "Hpca_section1.png",
    },
    {
        "title": "Hpca(section 2)",
        "gene": "Hpca",
        "section": "section2",
        "piece": "posterior",
        "filename": "Hpca_section2.png",
    },
]

POINT_SIZE = 2.05
CMAP = "viridis"


def read_10x_gene(h5_path, gene):
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        genes = np.array([x.decode("utf-8") for x in matrix["features"]["name"][:]])
        barcodes = np.array([x.decode("utf-8") for x in matrix["barcodes"][:]])

        hits = np.flatnonzero(genes == gene)
        if len(hits) == 0:
            raise ValueError(f"Gene {gene!r} was not found in {h5_path}")

        counts = sparse.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=tuple(matrix["shape"][:]),
        )
        expression = counts[hits[0], :].toarray().ravel().astype(float)

    return pd.DataFrame({"barcode": barcodes, "expression": expression})


def read_positions(path):
    names = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if first_line.startswith("barcode,"):
        positions = pd.read_csv(path)
    else:
        positions = pd.read_csv(path, header=None, names=names)

    positions = positions.loc[positions["in_tissue"].astype(int) == 1].copy()
    positions["x"] = positions["pxl_col_in_fullres"].astype(float)
    positions["y"] = positions["pxl_row_in_fullres"].astype(float)
    return positions[["barcode", "x", "y"]]


def normalize_piece(xy):
    xy = xy - xy.mean(axis=0, keepdims=True)
    xy[:, 1] = -xy[:, 1]
    xy[:, 0] -= xy[:, 0].min()
    xy[:, 1] -= xy[:, 1].min()
    return xy


def load_piece(h5_path, positions_path, gene):
    expr = read_10x_gene(h5_path, gene)
    positions = read_positions(positions_path)
    data = positions.merge(expr, on="barcode", how="inner")
    if data.empty:
        raise ValueError(f"No matching barcodes for {h5_path}")

    xy = normalize_piece(data[["x", "y"]].to_numpy(dtype=float))
    data["x_plot"] = xy[:, 0]
    data["y_plot"] = xy[:, 1]
    return data


def load_panel(section_cfg, piece, gene):
    if piece == "anterior":
        return load_piece(section_cfg["anterior_h5"], section_cfg["anterior_pos"], gene)
    if piece == "posterior":
        return load_piece(section_cfg["posterior_h5"], section_cfg["posterior_pos"], gene)
    raise ValueError(f"Unknown piece: {piece}")


def color_values(expression):
    values = np.log1p(expression.to_numpy(dtype=float))
    high = np.percentile(values, 99.2)
    if high <= 0:
        high = values.max() if values.max() > 0 else 1.0
    return np.clip(values, 0, high), high


def draw_panel(ax, data, title):
    colors, vmax = color_values(data["expression"])
    ax.scatter(
        data["x_plot"],
        data["y_plot"],
        c=colors,
        s=POINT_SIZE,
        cmap=CMAP,
        vmin=0,
        vmax=vmax,
        marker="o",
        linewidths=0,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=12, fontweight="normal", pad=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.01)


def save_single(panel, data):
    out_png = OUT_DIR / panel["filename"]
    fig, ax = plt.subplots(figsize=(2.05, 2.15), dpi=300)
    draw_panel(ax, data, panel["title"])
    fig.savefig(out_png, dpi=500, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out_png


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    sections = {"section1": SECTION_1, "section2": SECTION_2}
    cache = {}
    outputs = []

    for panel in PANELS:
        key = (panel["section"], panel["piece"], panel["gene"])
        if key not in cache:
            cache[key] = load_panel(
                sections[panel["section"]], panel["piece"], panel["gene"]
            )
        outputs.append(save_single(panel, cache[key]))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.25, 2.25), dpi=300)
    for ax, panel in zip(axes, PANELS):
        draw_panel(
            ax,
            cache[(panel["section"], panel["piece"], panel["gene"])],
            panel["title"],
        )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.18)
    montage = OUT_DIR / "gene_highlight_5panels.png"
    fig.savefig(montage, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    for path in outputs:
        print(f"Saved: {path}")
    print(f"Saved: {montage}")


if __name__ == "__main__":
    main()
