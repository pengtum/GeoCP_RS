"""
Generate: (1) per-tile full 5-panel figures for all 10 S2 tiles at 5 km scale;
          (2) a compact 10-tile overview grid showing RGB + q_hat_j per tile.

Output files:
    fig_s2_spatial_{tile}_s500.png          (all 10 tiles, 5-panel each)
    fig_s2_overview_all_tiles.png           (2 × 5 grid, RGB + q_hat per tile)
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

TILE_DIR = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                "My Drive/sentinel2_landcover_pilot_10m/tiles")
CKPT_DIR = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                "My Drive/s2_alpha_sweep/checkpoints")
PAPER_FIG = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                 "My Drive/Research/0_GeoCP_LISA/Manuscript/SACP_GeoCP_HSI/"
                 "Paper_GeoCP_Hyper/figures")
ANL_FIG = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
               "My Drive/Research/0_GeoCP_LISA/GeoCP_RS/analysis/figures")

ESA_COLORS = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff",
    50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8",
    90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
}
ESA_NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
             60: "bare", 70: "snow", 80: "water", 90: "wetland",
             95: "mangrove", 100: "lichen"}

TILES_ORDERED = [
    ("polk_iowa", "Polk, IA"),
    ("lancaster_pa", "Lancaster, PA"),
    ("hartford_ct", "Hartford, CT"),
    ("everglades_fl", "Everglades, FL"),
    ("lubbock_tx", "Lubbock, TX"),
    ("sacramento_ca", "Sacramento, CA"),
    ("phoenix_az", "Phoenix, AZ"),
    ("yellowstone_wy", "Yellowstone, WY"),
    ("seattle_wa", "Seattle, WA"),
    ("mississippi_la", "Mississippi, LA"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def load_tile(tile, scale_px=500):
    ck = CKPT_DIR / f"{tile}_s{scale_px}.pkl"
    d = pickle.load(open(ck, "rb"))
    full = np.load(TILE_DIR / f"{tile}.npz")
    emb_full, lab_full = full["emb"], full["label"]
    Hf, Wf = lab_full.shape
    r0 = Hf // 2 - scale_px // 2; c0 = Wf // 2 - scale_px // 2
    emb = emb_full[r0:r0+scale_px, c0:c0+scale_px]
    gt  = lab_full[r0:r0+scale_px, c0:c0+scale_px]
    return d, emb, gt


def rgb_from_emb(emb):
    rgb = emb[:, :, [3, 2, 1]].astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        out[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def scatter_to_grid(vals, idx, H, W, fill=np.nan):
    out = np.full(H * W, fill, dtype=np.float64)
    out[idx] = vals
    return out.reshape(H, W)


def per_tile_5panel(tile, nice, scale_px=500):
    """Five-panel detailed figure for one tile (RGB / GT / pred / size_C / q_C)."""
    d, emb, gt = load_tile(tile, scale_px)
    H, W = int(d["H"]), int(d["W"])
    test_idx = d["test_flat_idx"]
    pred_te  = d["pred_te"]
    sizes_C  = d["per_alpha"]["0.10"]["C"]["sizes"]
    q_C      = d["per_alpha"]["0.10"]["q_C"]
    # invert class remap from cal flat_idx + y_ca
    flat_lab = gt.ravel()
    cal_idx = d["cal_flat_idx"]
    y_ca_raw = flat_lab[cal_idx]
    y_ca_mod = d["y_ca"]
    code_of_mod = {int(m): int(c) for m, c in zip(y_ca_mod, y_ca_raw)}
    pred_raster = np.full(H * W, np.nan)
    pred_raster[test_idx] = [code_of_mod[int(m)] for m in pred_te]
    pred_raster = pred_raster.reshape(H, W)
    size_raster = scatter_to_grid(sizes_C, test_idx, H, W)
    q_raster    = scatter_to_grid(q_C,     test_idx, H, W)

    all_codes = sorted(set(gt.ravel().tolist()) - {0})
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    gt_cat = np.vectorize(lambda x: code_to_idx.get(x, -1))(gt).astype(float)
    gt_cat = np.where(gt == 0, np.nan, gt_cat)
    pred_cat = np.vectorize(lambda x: code_to_idx.get(int(x), -1) if not np.isnan(x) else np.nan)(pred_raster)
    cmap_cat = ListedColormap([ESA_COLORS.get(c, "#cccccc") for c in all_codes])

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    axes[0].imshow(rgb_from_emb(emb), interpolation="nearest")
    axes[0].set_title(f"S2 RGB — {nice}")
    axes[1].imshow(gt_cat, cmap=cmap_cat, vmin=-0.5, vmax=len(all_codes)-0.5, interpolation="nearest")
    axes[1].set_title(f"ESA GT ({len(all_codes)} classes)")
    axes[2].imshow(pred_cat, cmap=cmap_cat, vmin=-0.5, vmax=len(all_codes)-0.5, interpolation="nearest")
    axes[2].set_title(f"XGBoost pred. (acc={d['accuracy']:.2f})")
    vmax_sz = max(1, float(np.nanmax(size_raster)))
    im = axes[3].imshow(size_raster, cmap="viridis", vmin=0, vmax=vmax_sz, interpolation="nearest")
    axes[3].set_title(f"(C) set size (mean={float(np.nanmean(size_raster)):.2f})")
    plt.colorbar(im, ax=axes[3], fraction=0.045, pad=0.02)
    vmin_q, vmax_q = np.nanpercentile(q_C, [2, 98])
    im = axes[4].imshow(q_raster, cmap="magma", vmin=vmin_q, vmax=vmax_q, interpolation="nearest")
    axes[4].set_title(f"$\\hat q_j$ (sd={float(np.nanstd(q_raster)):.3f})")
    plt.colorbar(im, ax=axes[4], fraction=0.045, pad=0.02)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])

    leg = [plt.Rectangle((0,0),1,1,color=ESA_COLORS[c],label=f"{c}:{ESA_NAMES[c]}") for c in all_codes]
    fig.legend(handles=leg, loc="lower center", ncol=min(len(all_codes),8),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5,-0.02))
    fig.tight_layout()
    out = PAPER_FIG / f"fig_s2_spatial_{tile}_s{scale_px}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    # mirror
    for mirror_dir in [ANL_FIG,
        Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
             "My Drive/Research/0_GeoCP_LISA/Manuscript/SACP_GeoCP_HSI_Package/paper/figures")]:
        try: plt.imread(out)  # ensure file exists
        except: pass
        import shutil; shutil.copy2(out, mirror_dir / out.name)
    print("saved", out.name)


def overview_grid(scale_px=500):
    """Compact 10-tile overview: 5 rows × 4 cols (RGB | GT | size | q_j) — skip pred column."""
    n_tiles = len(TILES_ORDERED)
    fig, axes = plt.subplots(n_tiles, 4, figsize=(11, 2.1 * n_tiles))
    for i, (tile, nice) in enumerate(TILES_ORDERED):
        try:
            d, emb, gt = load_tile(tile, scale_px)
        except Exception as e:
            print(f"{tile}: {e}"); continue
        H, W = int(d["H"]), int(d["W"])
        test_idx = d["test_flat_idx"]
        sizes_C  = d["per_alpha"]["0.10"]["C"]["sizes"]
        q_C      = d["per_alpha"]["0.10"]["q_C"]
        size_raster = scatter_to_grid(sizes_C, test_idx, H, W)
        q_raster    = scatter_to_grid(q_C,     test_idx, H, W)
        all_codes = sorted(set(gt.ravel().tolist()) - {0})
        code_to_idx = {c: i for i, c in enumerate(all_codes)}
        gt_cat = np.vectorize(lambda x: code_to_idx.get(x, -1))(gt).astype(float)
        gt_cat = np.where(gt == 0, np.nan, gt_cat)
        cmap_cat = ListedColormap([ESA_COLORS.get(c, "#cccccc") for c in all_codes])

        axes[i, 0].imshow(rgb_from_emb(emb), interpolation="nearest")
        axes[i, 0].set_ylabel(nice, fontsize=10, rotation=90, labelpad=6)
        axes[i, 1].imshow(gt_cat, cmap=cmap_cat, vmin=-0.5, vmax=len(all_codes)-0.5, interpolation="nearest")
        vmax_sz = max(1, float(np.nanmax(size_raster)))
        im3 = axes[i, 2].imshow(size_raster, cmap="viridis", vmin=0, vmax=vmax_sz, interpolation="nearest")
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.045, pad=0.02)
        vmin_q, vmax_q = np.nanpercentile(q_C, [2, 98])
        im4 = axes[i, 3].imshow(q_raster, cmap="magma", vmin=vmin_q, vmax=vmax_q, interpolation="nearest")
        plt.colorbar(im4, ax=axes[i, 3], fraction=0.045, pad=0.02)

        # small per-tile annotation
        axes[i, 2].set_title(f"acc={d['accuracy']:.2f}  sz̄={float(np.nanmean(size_raster)):.2f}", fontsize=8)
        axes[i, 3].set_title(f"sd($\\hat q$)={float(np.nanstd(q_raster)):.3f}", fontsize=8)
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values(): spine.set_visible(False)

    axes[0, 0].set_title("S2 RGB (5 km)", fontsize=11)
    axes[0, 1].set_title("ESA WorldCover GT", fontsize=11)
    axes[0, 2].set_title("GeoCP-RS set size", fontsize=11)
    axes[0, 3].set_title(r"GeoCP-RS $\hat q_j$ field", fontsize=11)

    fig.tight_layout()
    out = PAPER_FIG / "fig_s2_overview_all_tiles.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    # mirror
    import shutil
    shutil.copy2(out, ANL_FIG / out.name)
    shutil.copy2(out, Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                           "My Drive/Research/0_GeoCP_LISA/Manuscript/"
                           "SACP_GeoCP_HSI_Package/paper/figures") / out.name)
    print("saved overview:", out.name)


if __name__ == "__main__":
    for tile, nice in TILES_ORDERED:
        try:
            per_tile_5panel(tile, nice, 500)
        except Exception as e:
            print(f"[per-tile] {tile} failed: {e}")
    overview_grid(500)
