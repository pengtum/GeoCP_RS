"""
Generate a 5-panel S2 spatial figure for the paper: RGB composite, ESA GT,
predicted class on test pixels, GeoCP-RS set size, q_j.
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
OUT = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
           "My Drive/Research/0_GeoCP_LISA/GeoCP_RS/analysis/figures")

ESA_COLORS = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff",
    50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8",
    90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
}
ESA_NAMES = {
    10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
    60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
    100: "lichen",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def rgb_from_emb(emb_patch):
    """Make an RGB composite from a 13-band S2 patch. Bands are (B1, B2, B3, B4, ...).
    RGB uses B4 (red, idx 3), B3 (green, idx 2), B2 (blue, idx 1). Per-channel
    2-98% percentile stretch for contrast."""
    rgb = emb_patch[:, :, [3, 2, 1]].astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        out[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def make_fig(tile, scale_px, seed_tile_emb_path):
    ck = CKPT_DIR / f"{tile}_s{scale_px}.pkl"
    d = pickle.load(open(ck, "rb"))
    H, W = int(d["H"]), int(d["W"])
    assert H == scale_px == W

    # Central crop of the raw tile matching the pkl label crop
    full = np.load(seed_tile_emb_path)
    emb_full = full["emb"]; lab_full = full["label"]
    Hf, Wf = lab_full.shape
    r0 = Hf // 2 - scale_px // 2
    c0 = Wf // 2 - scale_px // 2
    emb = emb_full[r0:r0+scale_px, c0:c0+scale_px]
    gt  = lab_full[r0:r0+scale_px, c0:c0+scale_px]

    # Grids: test pixels only
    test_idx = d["test_flat_idx"]
    pred_te  = d["pred_te"]
    sizes_C  = d["per_alpha"]["0.10"]["C"]["sizes"]
    q_C      = d["per_alpha"]["0.10"]["q_C"]

    # Reconstruct predicted-class map: remap model indices back to ESA codes
    # d['y_ca'] / d['y_te'] are remapped to 0..K-1. We need to invert the remap.
    classes_present = sorted(set(gt.ravel().tolist()) - {0})
    # Rare-class filter inside the pipeline may drop some codes; the classes
    # actually modeled are the set of codes that appear >= min_count.
    # Easiest way: derive the mapping from the label codes observed at cal/test flat indices.
    flat_label = gt.ravel()
    cal_idx = d["cal_flat_idx"]
    y_ca_raw = flat_label[cal_idx]   # ESA codes
    y_ca_mod = d["y_ca"]             # model indices
    code_of_mod = {}
    for m, c in zip(y_ca_mod, y_ca_raw):
        code_of_mod[int(m)] = int(c)

    pred_raster = np.full(H * W, np.nan, dtype=np.float64)
    pred_raster[test_idx] = [code_of_mod[int(m)] for m in pred_te]
    pred_raster = pred_raster.reshape(H, W)

    size_raster = np.full(H * W, np.nan, dtype=np.float64)
    size_raster[test_idx] = sizes_C
    size_raster = size_raster.reshape(H, W)

    q_raster = np.full(H * W, np.nan, dtype=np.float64)
    q_raster[test_idx] = q_C
    q_raster = q_raster.reshape(H, W)

    # Categorical cmap for ESA
    all_codes = sorted(set(gt.ravel().tolist()) - {0})
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    gt_cat = np.vectorize(lambda x: code_to_idx.get(x, -1))(gt).astype(float)
    gt_cat = np.where(gt == 0, np.nan, gt_cat)
    pred_cat = np.vectorize(lambda x: code_to_idx.get(x, -1) if not np.isnan(x) else np.nan)(pred_raster)
    cmap_cat = ListedColormap([ESA_COLORS.get(c, "#cccccc") for c in all_codes])

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.9))

    # Panel 1: RGB
    rgb = rgb_from_emb(emb)
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title(f"Sentinel-2 RGB\n{tile} @ {scale_px*0.01:.0f} km")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Panel 2: GT
    axes[1].imshow(gt_cat, cmap=cmap_cat, vmin=-0.5, vmax=len(all_codes)-0.5, interpolation="nearest")
    axes[1].set_title(f"ESA WorldCover GT\n({len(all_codes)} classes)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Panel 3: predicted class on test pixels
    axes[2].imshow(pred_cat, cmap=cmap_cat, vmin=-0.5, vmax=len(all_codes)-0.5, interpolation="nearest")
    axes[2].set_title(f"XGBoost prediction\n(test pixels, acc = {d['accuracy']:.2f})")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    # Panel 4: GeoCP-RS prediction-set size
    vmax_sz = max(1, np.nanmax(size_raster))
    im = axes[3].imshow(size_raster, cmap="viridis", vmin=0, vmax=vmax_sz, interpolation="nearest")
    axes[3].set_title(f"(C) GeoCP-RS set size\n(mean = {np.nanmean(size_raster):.2f})")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    plt.colorbar(im, ax=axes[3], fraction=0.045, pad=0.02)

    # Panel 5: q_j
    vmin_q, vmax_q = np.nanpercentile(q_C, [2, 98])
    im = axes[4].imshow(q_raster, cmap="magma", vmin=vmin_q, vmax=vmax_q, interpolation="nearest")
    axes[4].set_title(f"$\\hat q_j$ field (new output)\nsd = {np.nanstd(q_raster):.3f}")
    axes[4].set_xticks([]); axes[4].set_yticks([])
    plt.colorbar(im, ax=axes[4], fraction=0.045, pad=0.02)

    # ESA color legend (below panel 2)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=ESA_COLORS[c], label=f"{c}: {ESA_NAMES[c]}")
        for c in all_codes
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(all_codes), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_path = OUT / f"fig_s2_spatial_{tile}_s{scale_px}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    # Three representative tiles at 5 km scale
    for tile in ["sacramento_ca", "polk_iowa", "lubbock_tx"]:
        make_fig(tile, 500, TILE_DIR / f"{tile}.npz")
