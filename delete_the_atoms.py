#!/usr/bin/env python3
"""
'Delete the Atoms' -- Zero-Peak Residual Inversion experiment.

Pipeline
--------
1. Download the py4DSTEM tutorial 4D-STEM dataset from Google Drive.
2. Build the raw diffraction "splat-map" (mean diffraction pattern over all
   probe positions).
3. Digitally mask (black out) the bright Bragg peaks.
4. Isolate the faint Thermal Diffuse Scattering (TDS) background by subtracting
   a smooth median background from the masked splat-map.
5. Plot the isolated residual ("trash" signal) and quantify it.

Provenance notes
---------------
* py4DSTEM 0.14.18 does NOT expose `py4DSTEM.io.download_file_from_google_drive`.
  The real helper is `py4DSTEM.io.google_drive_downloader.gdrive_download`
  (aliased as `py4DSTEM.io.gdrive_download`).
* The requested Drive file ID 1PmbCYosA1eYydWmmZebvf6uon9k_5g_S is py4DSTEM's
  built-in "Au_sim" tutorial dataset. It downloads as `Au_sim.h5`, not
  `simulatedAuNanoplatelet_binned.h5` (that filename does not exist upstream).
* Diffraction-pattern center is assumed at the array center (true for this
  simulated dataset); stated explicitly wherever it matters.
* Bragg detection that survived: cross-correlate the max-projection with the
  file's own probe template, take local maxima above 1.2% of the correlation
  maximum (clean valley: 33 centers, then background specks). Mask R=8px for
  the direct beam (radial profile: 2404@r5 -> 218@r7 -> halo ~70@r9+),
  R=6px for the 32 Bragg disks. Earlier attempts (global percentile
  thresholds, flood-fill) are documented in the run log: thresholds missed
  faint outer disks; flood-fill bled into the diffuse halo (41% masked).
* "TDS" here is operational: diffuse background surviving Bragg masking +
  21px-median background subtraction on a multislice SIMULATION. Whether the
  simulation includes phonon scattering is not established from the file;
  treat the residual as the diffuse-scatter proxy, not a calibrated TDS map.
"""

# Run log (condensed):
#  run1: py4DSTEM gdrive_download() -> TypeError (fuzzy kwarg vs gdown 6.4.2);
#        fell back to gdown.download(id=...) directly. 525.7 MB downloaded.
#  run2: py4DSTEM.io.read() returns None on multi-object file; legacy v0.11
#        layout unreadable by 0.14.18 reader -> read uint16 datacube via h5py.
#  run3: percentile-threshold mask kept bg label 0 -> 100% masked (bug fixed).
#  run4: p98 threshold caught only 3 brightest regions -> residual dominated
#        by unmasked-Bragg leakage.
#  run5/6: p95 threshold, 11 regions, 14.5% masked; outer disks still missed.
#  run7: probe-template cross-correlation, bad threshold -> 1 center.
#  run8: threshold 1.2% of corr max -> 33 centers, R=7 fixed -> 31% masked.
#  run9: R=5 -> direct-beam skirt leaked (residual peak ~= Bragg peak).
#  run10: flood-fill per disk -> bled to 40.8% masked.
#  run11 (final): beam R=8 / disks R=6 from measured radial profiles.

import os
import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter

WORK = os.path.expanduser("~/workspace/tds_experiment")
os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------- 1. download
# NOTE: py4DSTEM 0.14.18's gdrive_download() passes fuzzy=True, which the
# installed gdown 6.4.2 no longer accepts -> TypeError. Workaround: call gdown
# directly with the same file ID the "Au_sim" key resolves to upstream.
import gdown  # noqa: E402

FILE_ID = "1PmbCYosA1eYydWmmZebvf6uon9k_5g_S"  # py4DSTEM "Au_sim" tutorial data
fpath = os.path.join(WORK, "Au_sim.h5")
if not os.path.exists(fpath):
    gdown.download(id=FILE_ID, output=fpath, quiet=False)
assert os.path.exists(fpath), f"download failed, not found: {fpath}"
print(f"[1] dataset on disk: {fpath} "
      f"({os.path.getsize(fpath) / 1e6:.1f} MB)", flush=True)

# ---------------------------------------------------------------- 2. load
# NOTE: Au_sim.h5 is a legacy py4DSTEM v0.11.10 file; py4DSTEM 0.14.18's reader
# cannot parse its internal layout (both bare-name and full-path datapaths
# fail), so the raw uint16 datacube is read directly with h5py. The array is
# a plain (Rx, Ry, Qx, Qy) stack -- no py4DSTEM-specific decoding needed for
# splat-map / masking / background-subtraction math.
import h5py  # noqa: E402

with h5py.File(fpath, "r") as f:
    dset = f["4DSTEM_experiment/data/datacubes/simulation_4DSTEM/data"]
    print(f"[2] dataset: shape {dset.shape}, dtype {dset.dtype}", flush=True)
    data = dset[...].astype(np.float32)  # (100, 84, 125, 125), ~524 MB
print(f"[2] datacube shape (Rx, Ry, Qx, Qy) = {data.shape}", flush=True)
rx, ry, qx, qy = data.shape

# ------------------------------------------------- 3. raw splat-map
mean_dp = data.mean(axis=(0, 1))   # average diffraction pattern ("splat-map")
max_dp = data.max(axis=(0, 1))     # max-projection: best peak finder
print(f"[3] splat-map total counts = {mean_dp.sum():.3e}", flush=True)

# ------------------------------------------------- 4. Bragg mask
# Standard 4D-STEM approach: cross-correlate the max-projection with the probe
# template stored in the file, take local maxima as Bragg disk centers, and
# mask a disk of the probe radius around each. This catches faint outer disks
# that a single global intensity threshold misses.
from scipy.signal import fftconvolve  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

with h5py.File(fpath, "r") as f:
    probe = f["4DSTEM_experiment/data/diffractionslices/probe_template/data"][
        ...
    ].astype(np.float64)
probe = np.clip(probe - probe.min(), 0, None)
pcy, pcx = (np.array(probe.shape) - 1) / 2.0
pyy, pxx = np.mgrid[0:probe.shape[0], 0:probe.shape[1]]
pr = np.hypot(pxx - pcx, pyy - pcy)
prof = np.array(
    [probe[(pr >= r) & (pr < r + 1)].mean() for r in range(int(pr.max()))]
)
below = np.where(prof < 0.5 * prof.max())[0]
R_probe = int(below[0]) if below.size else 6
print(f"[4] probe template {probe.shape}, half-max radius ~{R_probe}px",
      flush=True)

corr = fftconvolve(max_dp, probe[::-1, ::-1], mode="same")
foot = 2 * R_probe + 3
local_max = corr == maximum_filter(corr, size=foot)
# keep strong local maxima: the direct beam dominates (corr.max), while true
# Bragg disks sit at >=1.2% of it; background specks fall below (diagnosed a
# clean valley at ~1% on this dataset: 33 centers above, 3 specks below)
cthresh = 0.012 * corr.max()
centers = np.argwhere(local_max & (corr > cthresh))
print(f"[4] correlation threshold = {cthresh:.3e} "
      f"(max {corr.max():.3e}); disk centers found: {len(centers)}", flush=True)

# --- final mask: fixed radii, beam handled separately -----------------------
# Fixed radii beat flood-fill here (no bleed into the diffuse halo).
# Direct beam: R=8px (profile: 2404 @r5 -> 218 @r7 -> halo ~70 @r9+).
# Bragg disks: R=6px (probe half-max 5px + 1px margin; disks are ~15-20px
# apart so 6px radii don't merge).
yy, xx = np.mgrid[0:qy, 0:qx]
beam_idx = int(np.argmax([corr[yc, xc] for yc, xc in centers]))
mask = np.zeros_like(max_dp, dtype=bool)
for i, (yc, xc) in enumerate(centers):
    R = 8 if i == beam_idx else 6
    mask |= (xx - xc) ** 2 + (yy - yc) ** 2 <= R**2
n_peaks = len(centers)
print(f"[4] Bragg disks masked: {n_peaks} (beam R=8, disks R=6)  |  "
      f"masked pixel fraction: {100 * mask.mean():.2f}%", flush=True)

masked_dp = mean_dp.copy()
masked_dp[mask] = 0.0  # black out

# ------------------------------------------------- 5. TDS isolation
# Inpaint masked sites with the unmasked median, estimate the smooth
# background with a wide median filter, subtract -> diffuse residual.
inpaint_val = np.median(mean_dp[~mask])
tmp = mean_dp.copy()
tmp[mask] = inpaint_val
smooth_bg = median_filter(tmp, size=21)
tds = np.clip(mean_dp - smooth_bg, 0, None)
tds[mask] = 0.0  # Bragg sites stay black in the residual
tds_frac = 100 * tds.sum() / mean_dp.sum()
print(f"[5] isolated TDS counts = {tds.sum():.3e} "
      f"({tds_frac:.2f}% of splat-map)", flush=True)

# --- streak/anisotropy diagnostic: 8 azimuthal wedges in an annulus --------
cy, cx = qy / 2.0, qx / 2.0  # assumed pattern center (simulated data)
yy, xx = np.mgrid[0:qy, 0:qx]
rr = np.hypot(xx - cx, yy - cy)
phi = np.arctan2(yy - cy, xx - cx)
# annulus avoiding the direct beam and the detector edge
r_lo, r_hi = 0.15 * qx, 0.45 * qx
ann = (rr > r_lo) & (rr < r_hi) & (~mask) & (tds > 0)
wedge_means = []
for k in range(8):
    w = ann & (phi >= -np.pi + k * np.pi / 4) & (phi < -np.pi + (k + 1) * np.pi / 4)
    wedge_means.append(tds[w].mean() if w.any() else np.nan)
wedge_means = np.array(wedge_means)
aniso = np.nanmax(wedge_means) / np.nanmin(wedge_means)
print(f"[5] annulus {r_lo:.0f}-{r_hi:.0f} px | wedge means: "
      f"{np.round(wedge_means, 3)}", flush=True)
print(f"[5] azimuthal anisotropy (max/min wedge) = {aniso:.2f}", flush=True)
print(f"[5] TDS peak pixel = {tds.max():.3e} vs splat-map peak = "
      f"{mean_dp.max():.3e} (ratio 1:{mean_dp.max() / max(tds.max(), 1e-12):.0f})",
      flush=True)

# ---------------------------------------------------------------- 6. plots
fig, ax = plt.subplots(2, 2, figsize=(11, 10))
fig.suptitle("'Delete the Atoms' — Zero-Peak Residual Inversion  (Au_sim, "
             "py4DSTEM 0.14.18)", fontsize=12)

vmax = mean_dp.max()
vmin_log = max(np.percentile(mean_dp[mean_dp > 0], 0.5), 1e-9)
a = ax[0, 0].imshow(mean_dp, norm=LogNorm(vmin=vmin_log, vmax=vmax),
                    cmap="inferno", origin="lower")
ax[0, 0].set_title("(a) raw diffraction splat-map (log)")
plt.colorbar(a, ax=ax[0, 0], fraction=0.046, pad=0.04)

ax[0, 1].imshow(max_dp, norm=LogNorm(vmin=max(max_dp[max_dp > 0].min(), 1e-9),
                                     vmax=max_dp.max()), cmap="inferno",
                origin="lower")
# mask outline in cyan
from scipy.ndimage import binary_erosion  # noqa: E402
edge = mask ^ binary_erosion(mask)
ys, xs = np.nonzero(edge)
ax[0, 1].scatter(xs, ys, s=1, c="cyan", alpha=0.6)
ax[0, 1].set_title(f"(b) Bragg mask: {n_peaks} peaks, "
                   f"{100 * mask.mean():.1f}% of pixels")

b = ax[1, 0].imshow(masked_dp,
                    norm=LogNorm(vmin=vmin_log, vmax=vmax), cmap="inferno",
                    origin="lower")
ax[1, 0].set_title("(c) Bragg peaks blacked out (log)")
plt.colorbar(b, ax=ax[1, 0], fraction=0.046, pad=0.04)

pos = tds[tds > 0]
vmax_tds = np.log1p(np.percentile(pos, 99.5)) if pos.size else 1.0
c = ax[1, 1].imshow(np.log1p(tds), cmap="magma", origin="lower",
                    vmin=0, vmax=vmax_tds)
ax[1, 1].set_title(f"(d) isolated TDS residual — log1p "
                   f"({tds_frac:.2f}% of counts)")
plt.colorbar(c, ax=ax[1, 1], fraction=0.046, pad=0.04,
             label="log1p(counts)")

for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

fig.tight_layout()
out_png = os.path.join(WORK, "tds_residual.png")
fig.savefig(out_png, dpi=130)
print(f"[6] figure saved: {out_png}", flush=True)

# numeric residual for reuse
np.save(os.path.join(WORK, "tds_residual.npy"), tds)
np.save(os.path.join(WORK, "bragg_mask.npy"), mask)
print("[6] saved tds_residual.npy, bragg_mask.npy", flush=True)
print("DONE", flush=True)
