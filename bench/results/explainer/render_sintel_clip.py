"""Render 19 MPI-Sintel 'final' frames onto flyvis's 721-hexal eye (extent=15).

Uses flyvis's own rendering pipeline exactly as flyvis.datasets.sintel.RenderedSintel
does: BoxEye(extent=15, kernel_size=13) box-averages a center_crop_fraction=0.7 crop
of each grayscale frame at the hexal receptor centers. Writes:
  - sintel_clip.json  (uv coordinates + per-hexal luminance for 19 frames)
  - sintel_clip_check.png  (contact sheet: raw frames + hex-rendered frames)
"""
import os
import sys
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = str(HERE.parents[2])                          # repository root
SP = str(HERE)                                       # outputs land next to this script

# --- env / sys.path setup BEFORE importing flyvis ---
os.environ.setdefault("FLYVIS_ROOT_DIR", ROOT + "/flyvis_data")
sys.path[:0] = [ROOT + "/bench", ROOT + "/src"]
import m2_released  # noqa: E402  (side-effect free besides default FLYVIS_ROOT_DIR)
m2_released.patch_datamate_for_windows()

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

from flyvis.datasets.rendering import BoxEye
from flyvis.datasets.rendering.utils import split

SEQUENCE = "alley_2"
PASS_NAME = "final"
N_FRAMES = 19
EXTENT = 15
KERNEL_SIZE = 13
CENTER_CROP_FRACTION = 0.7
CLIP_DIR = f"{SP}/sintel_clip"          # raw PNG frames (not committed; fetched by HTTP range reads)
OUT_JSON = f"{SP}/sintel_clip.json"
OUT_PNG = f"{SP}/sintel_clip_check.png"


def sample_lum(path):
    """Identical to flyvis.datasets.sintel_utils.sample_lum."""
    return np.float32(Image.open(path).convert("L")) / 255


def receptor_uv(extent):
    """Reproduce BoxEye._receptor_centers()'s (u, v) enumeration order exactly
    (axial hex coords, pointy-ish box layout), WITHOUT the pixel-coordinate part.
    This is the ordering of the 721 hexals in every array BoxEye returns."""
    n = extent
    uv = []
    for u in range(-n, n + 1):
        v_min = max(-n, -n - u)
        v_max = min(n, n - u)
        for v in range(v_min, v_max + 1):
            uv.append((u, v))
    return uv


def main():
    frame_paths = sorted(glob.glob(os.path.join(CLIP_DIR, "frame_*.png")))
    assert len(frame_paths) == N_FRAMES, f"expected {N_FRAMES} frames, found {len(frame_paths)}"
    print(f"[render] {len(frame_paths)} frames from {CLIP_DIR}")

    raw_frames = [sample_lum(p) for p in frame_paths]  # each (H, W) float32 in [0,1]
    H, W = raw_frames[0].shape
    lum = torch.tensor(np.stack(raw_frames), dtype=torch.float32)  # (frames, H, W)
    print(f"[render] raw frame size: {W}x{H} (WxH), stacked lum shape {tuple(lum.shape)}")

    boxfilter = BoxEye(extent=EXTENT, kernel_size=KERNEL_SIZE)
    assert boxfilter.hexals == 721, boxfilter.hexals
    print(f"[render] BoxEye: hexals={boxfilter.hexals}, min_frame_size={boxfilter.min_frame_size.tolist()}")

    # Exactly flyvis.datasets.sintel.RenderedSintel's pipeline, with vertical_splits=1
    # (one rendering covering the full, center-cropped frame -- no left/right split).
    out_nelements = int(boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size)
    lum_split = split(lum, out_nelements, 1, CENTER_CROP_FRACTION)  # (1, frames, H, W_cropped)
    print(f"[render] after center_crop({CENTER_CROP_FRACTION}) + split(1): {tuple(lum_split.shape)}")

    lum_hex = boxfilter(lum_split, ftype="mean")  # (1, frames, 1, hexals)
    lum_hex = lum_hex[0, :, 0, :]  # (frames, hexals), float in [0,1]
    print(f"[render] hex-rendered shape: {tuple(lum_hex.shape)}, "
          f"range [{lum_hex.min():.3f}, {lum_hex.max():.3f}]")

    uv = receptor_uv(EXTENT)
    assert len(uv) == boxfilter.hexals == 721
    # Sanity-check our (u, v) reproduction against BoxEye's own receptor_centers y,x formula.
    # BoxEye casts (y, x) to torch.long (truncation toward zero, like a C cast),
    # so replicate that truncation before comparing.
    import math
    d = KERNEL_SIZE
    expected_yx = [(math.trunc(d * (u + v / 2)), math.trunc(d * v)) for u, v in uv]
    actual_yx = boxfilter.receptor_centers.tolist()
    max_err = max(abs(ey - ay) + abs(ex - ax) for (ey, ex), (ay, ax) in zip(expected_yx, actual_yx))
    print(f"[render] (u,v) ordering check vs BoxEye.receptor_centers: max abs err = {max_err}")
    assert max_err == 0, "uv ordering does not match BoxEye's receptor_centers order"

    lum_uint8 = (lum_hex.clamp(0, 1) * 255).round().to(torch.int64)  # (frames, hexals)
    lum_list = lum_uint8.tolist()

    payload = {
        "sequence": SEQUENCE,
        "pass": PASS_NAME,
        "frames": N_FRAMES,
        "hexals": boxfilter.hexals,
        "extent": EXTENT,
        "kernel_size": KERNEL_SIZE,
        "center_crop_fraction": CENTER_CROP_FRACTION,
        "source_frame_size_px": {"width": W, "height": H},
        "renderer": (
            "flyvis.datasets.rendering.BoxEye(extent=15, kernel_size=13) applied via "
            "flyvis.datasets.rendering.utils.split(out_nelements=min_frame_size[1]+2*kernel_size, "
            "n_splits=1, center_crop_fraction=0.7) then boxfilter(..., ftype='mean') -- the same "
            "pipeline flyvis.datasets.sintel.RenderedSintel uses to build MultiTaskSintel's lum "
            "channel (vertical_splits fixed to 1 here: a single rendering of the whole cropped "
            "frame, not flyvis's default 3-way left/mid/right split). Grayscale via "
            "PIL.Image.open(...).convert('L') (flyvis.datasets.sintel_utils.sample_lum), matching "
            "flyvis exactly."
        ),
        "uv_ordering": (
            "uv[i] = (u, v) axial hex coordinates of hexal i, in exactly the order "
            "BoxEye._receptor_centers() yields them: for u in range(-extent, extent+1): "
            "for v in range(max(-extent, -extent-u), min(extent, extent-u)+1): yield (u, v). "
            "This is the same order as every array BoxEye/RenderedSintel/MultiTaskSintel "
            "returns along its hexal axis (constraint: max(|u|,|v|,|u+v|) <= extent)."
        ),
        "uv": [[u, v] for u, v in uv],
        "lum": lum_list,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f)
    sz = os.path.getsize(OUT_JSON)
    print(f"[render] wrote {OUT_JSON} ({sz} bytes)")

    # ----- contact sheet -----------------------------------------------
    show_idx = [0, 4, 9, 14, 18]  # frames 1, 5, 10, 15, 19 (0-indexed)
    n_cols = len(show_idx)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6.4))

    for col, fi in enumerate(show_idx):
        ax = axes[0, col]
        ax.imshow(raw_frames[fi], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"raw frame {fi + 1:02d}", fontsize=9)
        ax.axis("off")

    # hex plot: axial (u,v) -> cartesian, pointy-ish layout matching BoxEye's own
    # (y = u + v/2, x = v), flipped/scaled for a natural on-screen look.
    us = np.array([u for u, v in uv], dtype=float)
    vs = np.array([v for u, v in uv], dtype=float)
    hex_x = vs
    hex_y = -(us + vs / 2.0)
    # circumradius so neighboring hexals touch: horizontal spacing between
    # same-row centers is 1 (since dx=1 per v step), so radius ~ 1/sqrt(3)
    hex_r = 1.0 / np.sqrt(3)

    lum_arr = np.array(lum_list, dtype=float) / 255.0  # (frames, hexals)
    for col, fi in enumerate(show_idx):
        ax = axes[1, col]
        vals = lum_arr[fi]
        colors = plt.cm.gray(vals)
        for x, y, c in zip(hex_x, hex_y, colors):
            ax.add_patch(plt.Circle((x, y), hex_r * 0.95, facecolor=c, edgecolor="none"))
        ax.set_xlim(hex_x.min() - 1, hex_x.max() + 1)
        ax.set_ylim(hex_y.min() - 1, hex_y.max() + 1)
        ax.set_aspect("equal")
        ax.set_facecolor("0.5")
        ax.set_title(f"hex frame {fi + 1:02d} (721 hexals)", fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"MPI-Sintel training/final/{SEQUENCE} -> flyvis 721-hexal eye "
        f"(extent={EXTENT}, kernel={KERNEL_SIZE}, center_crop={CENTER_CROP_FRACTION})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[render] wrote {OUT_PNG} ({os.path.getsize(OUT_PNG)} bytes)")


if __name__ == "__main__":
    main()
