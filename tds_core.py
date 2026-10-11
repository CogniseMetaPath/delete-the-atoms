#!/usr/bin/env python3
"""tds_core: 'Delete the Atoms' pipeline as importable functions.

Single source of truth for the batch script and the Streamlit app.
Logic mirrors delete_the_atoms.py run11 exactly; parameters exposed.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, median_filter


def splat_maps(h5path, dkey, rows=None):
    """Mean and max diffraction patterns without loading the full cube.

    h5path: HDF5 file; dkey: datacube path inside it.
    rows: optional (lo, hi) slice over Rx for split-half tests.
    """
    import h5py
    with h5py.File(h5path, "r") as f:
        dset = f[dkey]
        n_rx, n_ry = dset.shape[0], dset.shape[1]
        lo, hi = (0, n_rx) if rows is None else rows
        acc_sum, acc_max = None, None
        for i in range(lo, hi):
            chunk = dset[i].astype(np.float64)
            s, m = chunk.sum(axis=0), chunk.max(axis=0)
            acc_sum = s if acc_sum is None else acc_sum + s
            acc_max = m if acc_max is None else np.maximum(acc_max, m)
        npos = (hi - lo) * n_ry
    return (acc_sum / npos).astype(np.float32), acc_max.astype(np.float32)


def find_disks(max_dp, probe, thresh_frac=0.012):
    """Bragg disk centers via probe-template cross-correlation."""
    probe = np.clip(probe - probe.min(), 0, None).astype(np.float64)
    pcy, pcx = (np.array(probe.shape) - 1) / 2.0
    pyy, pxx = np.mgrid[0:probe.shape[0], 0:probe.shape[1]]
    pr = np.hypot(pxx - pcx, pyy - pcy)
    prof = np.array([probe[(pr >= r) & (pr < r + 1)].mean()
                     for r in range(int(pr.max()))])
    below = np.where(prof < 0.5 * prof.max())[0]
    r_probe = int(below[0]) if below.size else 6
    corr = fftconvolve(max_dp, probe[::-1, ::-1], mode="same")
    foot = 2 * r_probe + 3
    local_max = corr == maximum_filter(corr, size=foot)
    centers = np.argwhere(local_max & (corr > thresh_frac * corr.max()))
    beam_idx = int(np.argmax([corr[yc, xc] for yc, xc in centers]))
    return centers, beam_idx, r_probe


def build_mask(shape, centers, beam_idx, r_beam=8, r_disk=6):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    mask = np.zeros(shape, dtype=bool)
    for i, (yc, xc) in enumerate(centers):
        r = r_beam if i == beam_idx else r_disk
        mask |= (xx - xc) ** 2 + (yy - yc) ** 2 <= r ** 2
    return mask


def isolate_residual(mean_dp, mask, median_size=21):
    inpaint_val = np.median(mean_dp[~mask])
    tmp = mean_dp.copy()
    tmp[mask] = inpaint_val
    smooth_bg = median_filter(tmp, size=median_size)
    tds = np.clip(mean_dp - smooth_bg, 0, None)
    tds[mask] = 0.0
    return tds


def anisotropy(tds, mask, r_lo_frac=0.15, r_hi_frac=0.45, n_wedges=8,
               edge_exclusion_px=0):
    """Azimuthal max/min wedge ratio in an annulus around pattern center."""
    from scipy.ndimage import distance_transform_edt
    qy, qx = tds.shape
    cy, cx = qy / 2.0, qx / 2.0
    yy, xx = np.mgrid[0:qy, 0:qx]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    r_lo, r_hi = r_lo_frac * qx, r_hi_frac * qx
    ann = (rr > r_lo) & (rr < r_hi) & (~mask) & (tds > 0)
    if edge_exclusion_px > 0:
        ann &= distance_transform_edt(~mask) > edge_exclusion_px
    wedge_means = []
    for k in range(n_wedges):
        w = ann & (phi >= -np.pi + k * 2 * np.pi / n_wedges) & \
            (phi < -np.pi + (k + 1) * 2 * np.pi / n_wedges)
        wedge_means.append(tds[w].mean() if w.any() else np.nan)
    wedge_means = np.array(wedge_means)
    return np.nanmax(wedge_means) / np.nanmin(wedge_means), wedge_means


def fit_ellipticity(centers, beam_idx, shape):
    """Fit ellipse to disk radii vs angle; returns ellipticity estimate.

    For a circular pattern, disks at a given nominal radius have constant
    radius vs azimuthal angle. Elliptical distortion -> sinusoidal modulation.
    Returns (ellipticity, angle_of_major_axis_deg) from the strongest ring.
    """
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    pts = np.array([(xc - cx, yc - cy) for i, (yc, xc) in enumerate(centers)
                    if i != beam_idx])
    r = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    # group into rings by radius (tolerance 3px)
    order = np.argsort(r)
    r, ang = r[order], ang[order]
    rings, cur = [], [0]
    for i in range(1, len(r)):
        if r[i] - r[cur[-1]] < 3.0:
            cur.append(i)
        else:
            rings.append(cur)
            cur = [i]
    rings.append(cur)
    best = None
    for ring in rings:
        if len(ring) < 6:
            continue
        rr, aa = r[ring], ang[ring]
        # fit r(a) = r0 * (1 + e*cos(2(a - a0))) linearized
        A = np.column_stack([np.ones_like(aa), np.cos(2 * aa), np.sin(2 * aa)])
        coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
        r0, c, s = coef
        e = np.hypot(c, s) / r0
        if best is None or len(ring) > best[2]:
            best = (e, np.degrees(0.5 * np.arctan2(s, c)) % 180, len(ring), r0)
    return best  # (ellipticity, major_axis_deg, n_disks, mean_radius)


def run_pipeline(h5path, dkey, rows=None, thresh_frac=0.012,
                 r_beam=8, r_disk=6, median_size=21,
                 r_lo_frac=0.15, r_hi_frac=0.45, n_wedges=8,
                 probe=None):
    mean_dp, max_dp = splat_maps(h5path, dkey, rows=rows)
    if probe is None:
        import h5py
        with h5py.File(h5path, "r") as f:
            probe = f["4DSTEM_experiment/data/diffractionslices/"
                      "probe_template/data"][...].astype(np.float32)
    centers, beam_idx, r_probe = find_disks(max_dp, probe, thresh_frac)
    mask = build_mask(mean_dp.shape, centers, beam_idx, r_beam, r_disk)
    tds = isolate_residual(mean_dp, mask, median_size)
    aniso, wedges = anisotropy(tds, mask, r_lo_frac, r_hi_frac, n_wedges)
    tds_frac = 100 * tds.sum() / mean_dp.sum()
    ell = fit_ellipticity(centers, beam_idx, mean_dp.shape)
    return {
        "n_centers": len(centers),
        "masked_pct": 100 * mask.mean(),
        "tds_frac_pct": tds_frac,
        "anisotropy": aniso,
        "wedge_means": wedges,
        "peak_ratio": mean_dp.max() / max(tds.max(), 1e-12),
        "ellipticity": ell,
        "splat_total": float(mean_dp.sum()),
    }
