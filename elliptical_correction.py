#!/usr/bin/env python3
"""Proper elliptical correction for 'Delete the Atoms'.

Mechanism hypothesis: the diffraction pattern carries a real elliptical
distortion (measured ~6%). Circular masks on elliptical Bragg disks leave
crescent-shaped unmasked regions aligned with the major axis -> the
"azimuthal anisotropy" may be mask/disk shape mismatch, not physics.

Ellipse model (geometric, joint least-squares across rings):
    r(theta) = r0_k * (1 + e*cos(2*(theta - th0))), center (cx, cy) global.
Fit on ALL detected disk centers simultaneously (rings share e, th0, center).

Method (a) — no resampling, tests the mechanism directly:
  elliptical masks (same e, theta) + elliptical annulus/wedges.
Method (b) — conventional cross-check:
  dewarp the image with the fitted global ellipse (matrix inverse),
  verify circularization (re-fit ~ 0), re-run the standard pipeline.

Also ablates mask-shape vs annulus-geometry contributions.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from tds_core import (find_disks, build_mask, isolate_residual, anisotropy,
                      splat_maps)


# ---------------------------------------------------------- ellipse fitting
def _ring_groups(pts, cx0, cy0, tol=4.0, min_n=6):
    r = np.hypot(pts[:, 0]-cx0, pts[:, 1]-cy0)
    order = np.argsort(r)
    rs, ps = r[order], pts[order]
    groups, cur = [], [0]
    for i in range(1, len(rs)):
        if rs[i]-rs[cur[-1]] < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [ps[g] for g in groups if len(g) >= min_n]


def fit_global_ellipse(centers, beam_idx, shape, verbose=True):
    """Joint geometric least-squares fit over all rings.

    Returns (e, theta, cx, cy). Points are (x=col, y=row)."""
    cx0, cy0 = shape[1]/2.0, shape[0]/2.0
    pts = np.array([(xc, yc) for i, (yc, xc) in enumerate(centers)
                    if i != beam_idx])
    groups = _ring_groups(pts, cx0, cy0)
    K = len(groups)
    assert K >= 1, "no rings with enough disks"
    p0 = [cx0, cy0, 0.05, 0.0] + \
         [np.hypot(g[:, 0]-cx0, g[:, 1]-cy0).mean() for g in groups]
    lb = [cx0-10, cy0-10, -0.30, -np.pi] + [1.0]*K
    ub = [cx0+10, cy0+10,  0.30,  np.pi] + [200.0]*K

    def resid(p):
        cx, cy, e, th0 = p[0], p[1], p[2], p[3]
        out = []
        for k, g in enumerate(groups):
            r0 = p[4+k]
            dx, dy = g[:, 0]-cx, g[:, 1]-cy
            rr = np.hypot(dx, dy)
            th = np.arctan2(dy, dx)
            out.append(rr - r0*(1 + e*np.cos(2*(th-th0))))
        return np.concatenate(out)

    sol = least_squares(resid, p0, bounds=(lb, ub), method="trf")
    cx, cy, e, th0 = sol.x[0], sol.x[1], sol.x[2], sol.x[3]
    # normalize degeneracy: (e,th0) == (-e,th0+90deg); enforce e >= 0
    if e < 0:
        e, th0 = -e, th0 + np.pi/2
    th0 = th0 % np.pi
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    if verbose:
        print(f"joint fit: e={e:.4f} theta={np.degrees(th0):.1f}deg "
              f"center=({cx:.2f},{cy:.2f}) rms={rms:.3f}px "
              f"({sol.nfev} evals, {len(sol.fun)} pts, {K} rings)")
        # per-ring consistency: e_k from sinusoid with global center
        print("per-ring check (n, r0, e_k, theta_k):")
        for g in groups:
            dx, dy = g[:, 0]-cx, g[:, 1]-cy
            rr = np.hypot(dx, dy)
            th = np.arctan2(dy, dx)
            A = np.column_stack([np.ones_like(th), np.cos(2*th),
                                 np.sin(2*th)])
            c, *_ = np.linalg.lstsq(A, rr, rcond=None)
            ek = np.hypot(c[1], c[2])/c[0]
            tk = np.degrees(0.5*np.arctan2(c[2], c[1])) % 180
            print(f"  n={len(g):2d} r0={c[0]:6.1f} e={ek:.4f} "
                  f"theta={tk:6.1f}")
    return e, th0 % np.pi, cx, cy, rms


def self_test():
    rng = np.random.default_rng(0)
    cx, cy, e, th0 = 63.1, 61.8, 0.064, np.deg2rad(92.5)
    pts = []
    for r0, n in ((22.7, 8), (42.5, 10), (61.6, 14)):
        angs = rng.uniform(0, 2*np.pi, n)
        rr = r0*(1 + e*np.cos(2*(angs-th0)))
        x = cx + rr*np.cos(angs) + rng.normal(0, .3, n)
        y = cy + rr*np.sin(angs) + rng.normal(0, .3, n)
        pts.append(np.column_stack([x, y]))
    pts = np.vstack(pts)
    centers = [(0, 0)] + [(float(y), float(x)) for x, y in pts]  # (row,col)
    e2, th2, cx2, cy2, rms = fit_global_ellipse(centers, 0, (125, 125),
                                                verbose=False)
    assert abs(e2-e) < 0.01, (e2, e)
    assert abs(cx2-cx) < 0.5 and abs(cy2-cy) < 0.5
    dth = abs((th2-th0+np.pi/2) % np.pi - np.pi/2)
    assert dth < 0.1, dth
    print(f"self-test PASS: e={e2:.4f} (true {e}), "
          f"theta={np.degrees(th2):.1f}deg, center=({cx2:.2f},{cy2:.2f})")


# ------------------------------------------------- method (a): elliptical masks
def build_elliptical_mask(shape, centers, beam_idx, e, theta,
                          r_beam=8, r_disk=6, margin=1.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    ct, st = np.cos(theta), np.sin(theta)
    mask = np.zeros(shape, bool)
    for i, (yc, xc) in enumerate(centers):
        R = r_beam if i == beam_idx else r_disk
        am, bm = R*(1+e)+margin, R*(1-e)+margin
        dx, dy = xx-xc, yy-yc
        u = (dx*ct + dy*st)/am
        v = (-dx*st + dy*ct)/bm
        mask |= (u*u + v*v) <= 1
    return mask


def anisotropy_elliptical(tds, mask, cx, cy, e, theta,
                          r_lo_frac=0.15, r_hi_frac=0.45, n_wedges=8):
    """Wedge anisotropy in circularized coords (no image resampling)."""
    qy, qx = tds.shape
    yy, xx = np.mgrid[0:qy, 0:qx]
    dx, dy = xx-cx, yy-cy
    ct, st = np.cos(theta), np.sin(theta)
    xp = (dx*ct + dy*st)/(1+e)
    yp = (-dx*st + dy*ct)/(1-e)
    rr = np.hypot(xp, yp)
    phi = np.arctan2(yp, xp)
    r_lo, r_hi = r_lo_frac*qx, r_hi_frac*qx
    ann = (rr > r_lo) & (rr < r_hi) & (~mask) & (tds > 0)
    wm = []
    for k in range(n_wedges):
        w = ann & (phi >= -np.pi+k*2*np.pi/n_wedges) & \
            (phi < -np.pi+(k+1)*2*np.pi/n_wedges)
        wm.append(tds[w].mean() if w.any() else np.nan)
    wm = np.array(wm)
    return np.nanmax(wm)/np.nanmin(wm), wm


# ------------------------------------------------- method (b): dewarp
def dewarp(img, cx, cy, e, theta, order=3):
    qy, qx = img.shape
    yy, xx = np.mgrid[0:qy, 0:qx]
    dx, dy = xx-cx, yy-cy
    ct, st = np.cos(theta), np.sin(theta)
    u = dx*ct + dy*st
    v = -dx*st + dy*ct
    ui, vi = u*(1+e), v*(1-e)
    dxi = ui*ct - vi*st
    dyi = ui*st + vi*ct
    return map_coordinates(img, [cy+dyi, cx+dxi], order=order, mode="nearest")


def main():
    self_test()
    H = "Au_sim.h5"
    K = "4DSTEM_experiment/data/datacubes/simulation_4DSTEM/data"
    mean_dp, max_dp = splat_maps(H, K)
    import h5py
    with h5py.File(H, "r") as f:
        probe = f["4DSTEM_experiment/data/diffractionslices/"
                  "probe_template/data"][...].astype(np.float32)

    centers, beam_idx, _ = find_disks(max_dp, probe)
    e, theta, ecx, ecy, rms = fit_global_ellipse(centers, beam_idx,
                                                 mean_dp.shape)

    print("\n--- ablation table (anisotropy) ---")
    m0 = build_mask(mean_dp.shape, centers, beam_idx)
    t0 = isolate_residual(mean_dp, m0)
    a0, _ = anisotropy(t0, m0)
    print(f"1. circular masks + circular annulus (baseline): {a0:.3f}")

    m1 = build_elliptical_mask(mean_dp.shape, centers, beam_idx, e, theta)
    t1 = isolate_residual(mean_dp, m1)
    a1, _ = anisotropy(t1, m1)
    print(f"2. elliptical masks + circular annulus:            {a1:.3f} "
          f"(masked {m1.mean()*100:.2f}%, tds {100*t1.sum()/mean_dp.sum():.2f}%)")

    a2, _ = anisotropy_elliptical(t0, m0, ecx, ecy, e, theta)
    print(f"3. circular masks + elliptical annulus:            {a2:.3f}")

    a3, w3 = anisotropy_elliptical(t1, m1, ecx, ecy, e, theta)
    print(f"4. elliptical masks + elliptical annulus (proper): {a3:.3f}")

    md = dewarp(mean_dp.astype(np.float64), ecx, ecy, e, theta)
    xd = dewarp(max_dp.astype(np.float64), ecx, ecy, e, theta)
    c2, b2, _ = find_disks(xd.astype(np.float32), probe)
    e2, _, _, _, _ = fit_global_ellipse(c2, b2, md.shape, verbose=False)
    print(f"5. dewarped: residual ellipticity {e2:.4f} (was {e:.4f}), "
          f"n_centers={len(c2)}")
    m5 = build_mask(md.shape, c2, b2)
    t5 = isolate_residual(md.astype(np.float32), m5)
    a5, _ = anisotropy(t5, m5)
    print(f"   dewarped pipeline anisotropy: {a5:.3f} "
          f"(tds {100*t5.sum()/md.sum():.2f}%)")
    print("DONE")


if __name__ == "__main__":
    main()
