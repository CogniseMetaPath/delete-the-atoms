"""'Delete the Atoms' v2 -- the instrument, honestly presented.

An interactive instrument for interrogating what remains when you black out
the Bragg peaks in 4D-STEM data. Every knob re-runs the real pipeline on the
real data; every confound we found is documented on the page.

Run with:  streamlit run streamlit_app.py
Needs:     tds_core.py, elliptical_correction.py, assets/*.npy in this dir.
"""
from pathlib import Path

import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tds_core import find_disks, build_mask, isolate_residual, anisotropy
from elliptical_correction import (build_elliptical_mask,
                                   anisotropy_elliptical)

HERE = Path(__file__).parent

# Elliptical distortion: joint least-squares fit across all detected rings
# (elliptical_correction.py). Deterministic from the data; embedded so the
# app starts instantly.
ELL_E = 0.0729
ELL_THETA = np.deg2rad(94.3)
ELL_CX, ELL_CY = 62.71, 63.12

st.set_page_config(
    page_title="Delete the Atoms — an honest instrument",
    layout="wide",
)


@st.cache_data
def load_assets():
    d = HERE / "assets"
    return (np.load(d / "mean_dp.npy"), np.load(d / "max_dp.npy"),
            np.load(d / "probe.npy"))


@st.cache_data
def detect_centers(_max_dp, _probe, thresh):
    centers, beam_idx, _ = find_disks(_max_dp, _probe, thresh_frac=thresh)
    return centers, beam_idx


def make_figure(mean_dp, max_dp, mask, tds, title_note=""):
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle("'Delete the Atoms' — Zero-Peak Residual Inversion  "
                 "(Au_sim, py4DSTEM 0.14.18)" + title_note, fontsize=12)
    vmax = mean_dp.max()
    vmin_log = max(np.percentile(mean_dp[mean_dp > 0], 0.5), 1e-9)
    a = ax[0, 0].imshow(mean_dp, norm=LogNorm(vmin=vmin_log, vmax=vmax),
                        cmap="inferno", origin="lower")
    ax[0, 0].set_title("(a) raw diffraction splat-map (log)")
    plt.colorbar(a, ax=ax[0, 0], fraction=0.046, pad=0.04)

    from scipy.ndimage import binary_erosion
    ax[0, 1].imshow(max_dp,
                    norm=LogNorm(vmin=max(max_dp[max_dp > 0].min(), 1e-9),
                                 vmax=max_dp.max()),
                    cmap="inferno", origin="lower")
    edge = mask ^ binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    ax[0, 1].scatter(xs, ys, s=1, c="cyan", alpha=0.6)
    ax[0, 1].set_title(f"(b) Bragg mask: {int(mask.sum())} px masked "
                       f"({100*mask.mean():.1f}% of pixels)")

    b = ax[1, 0].imshow(np.where(mask, np.nan, mean_dp),
                        norm=LogNorm(vmin=vmin_log, vmax=vmax), cmap="inferno",
                        origin="lower")
    ax[1, 0].set_title("(c) Bragg peaks blacked out (log)")
    plt.colorbar(b, ax=ax[1, 0], fraction=0.046, pad=0.04)

    tds_frac = 100*tds.sum()/mean_dp.sum()
    pos = tds[tds > 0]
    vmax_tds = np.log1p(np.percentile(pos, 99.5)) if pos.size else 1.0
    c = ax[1, 1].imshow(np.log1p(tds), cmap="magma", origin="lower",
                        vmin=0, vmax=vmax_tds)
    ax[1, 1].set_title(f"(d) isolated diffuse residual — log1p "
                       f"({tds_frac:.2f}% of counts)")
    plt.colorbar(c, ax=ax[1, 1], fraction=0.046, pad=0.04,
                 label="log1p(counts)")
    for axi in ax.ravel():
        axi.set_xticks([])
        axi.set_yticks([])
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ header
st.title("'Delete the Atoms'")
st.caption("An instrument for interrogating what remains when you black out "
           "the Bragg peaks in 4D-STEM data — presented honestly, confounds "
           "included.")

st.info(
    "**Read this in 30 seconds.** Average a 4D-STEM scan into one diffraction "
    "pattern. Find every Bragg disk. Black them out — delete the atoms. What "
    "remains is the diffuse background: the whispers between the shouts. We "
    "built the instrument to measure it, then spent a week trying to prove "
    "it was fooling us. Twice it nearly did. This page shows the instrument, "
    "the near-misses, and the live knobs so you can check our work yourself."
)

mean_dp, max_dp, probe = load_assets()

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Diffuse residual (baseline)", "2.10%",
              "of splat-map counts survive Bragg removal")
with m2:
    st.metric("Azimuthal anisotropy (baseline)", "1.79",
              "max/min over 8 wedges, 19–56px annulus")
with m3:
    st.metric("After elliptical correction", "2.00 – 2.06",
              "two independent methods agree")

st.warning(
    "**Honest label.** The test dataset (py4DSTEM's `Au_sim`) is a "
    "calibration phantom — it very likely contains no thermal (phonon) "
    "scattering at all, and it deliberately includes elliptical distortion. "
    "So 'TDS' on this page means *diffuse background after Bragg removal*, "
    "an unvalidated proxy. The instrument is real and reproducible; the "
    "physical interpretation is pending. Details in 'The full story' below."
)

st.markdown("---")
st.header("Run the instrument yourself")
st.caption("Every knob re-runs the actual pipeline on the real data. "
           "Watch the headline numbers move — that is the point.")

cc1, cc2 = st.columns([1, 2])
with cc1:
    thresh = st.slider("Disk detection threshold", 0.005, 0.030, 0.012,
                       0.001, format="%.3f",
                       help="Fraction of max correlation kept as Bragg disks. "
                            "Lower finds fainter disks — and more specks.")
    r_beam = st.slider("Direct-beam mask radius (px)", 5, 12, 8, 1)
    r_disk = st.slider("Bragg-disk mask radius (px)", 4, 9, 6, 1)
    med = st.slider("Median background size (px)", 11, 41, 21, 2,
                    help="Odd sizes behave best. Too small eats real "
                         "structure; too large leaves gradients.")
    use_ell = st.checkbox("Elliptical correction (measured 7.3%)", True,
                          help="Use elliptical masks + elliptical annulus "
                               "from the joint distortion fit. Uncheck to "
                               "see the uncorrected baseline.")
    edge_excl = st.slider("Exclude px near mask edges", 0, 5, 0, 1,
                          help="Drops residual pixels within N px of any "
                               "mask edge from the anisotropy metric.")
    n_wedges = st.slider("Azimuthal wedges", 4, 16, 8, 1)

with cc2:
    centers, beam_idx = detect_centers(max_dp, probe, thresh)
    if use_ell:
        mask = build_elliptical_mask(mean_dp.shape, centers, beam_idx,
                                     ELL_E, ELL_THETA,
                                     r_beam=r_beam, r_disk=r_disk)
        tds = isolate_residual(mean_dp, mask, median_size=med)
        aniso, wedge_means = anisotropy_elliptical(
            tds, mask, ELL_CX, ELL_CY, ELL_E, ELL_THETA,
            n_wedges=n_wedges)
        if edge_excl:
            from scipy.ndimage import distance_transform_edt
            far = distance_transform_edt(~mask) > edge_excl
            aniso, wedge_means = anisotropy_elliptical(
                np.where(far, tds, 0.0), mask | ~far,
                ELL_CX, ELL_CY, ELL_E, ELL_THETA, n_wedges=n_wedges)
    else:
        mask = build_mask(mean_dp.shape, centers, beam_idx,
                          r_beam=r_beam, r_disk=r_disk)
        tds = isolate_residual(mean_dp, mask, median_size=med)
        if edge_excl:
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~mask)
            sel = dist > edge_excl
            tds_m = np.where(sel, tds, 0.0)
            aniso, wedge_means = anisotropy(
                tds_m, mask | ~sel, n_wedges=n_wedges)
        else:
            aniso, wedge_means = anisotropy(tds, mask, n_wedges=n_wedges)

    tds_frac = 100*tds.sum()/mean_dp.sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Disks found", f"{len(centers)}")
    k2.metric("Masked", f"{100*mask.mean():.1f}%")
    k3.metric("Residual", f"{tds_frac:.2f}%")
    k4.metric("Anisotropy", f"{aniso:.2f}")
    st.pyplot(make_figure(mean_dp, max_dp, mask, tds,
                          title_note="  — live lab"))
    st.caption("Wedge means (the anisotropy metric, max/min):")
    st.bar_chart(wedge_means)
st.markdown("---")
st.header("Fool the instrument")
st.caption("A synthetic diffraction pattern where you know the ground truth. "
           "Inject elliptical distortion into perfectly isotropic data and "
           "watch the instrument report anisotropy that isn't there — then "
           "correct it and watch it vanish. Masks are placed at the true "
           "disk positions, so this isolates the mask-shape effect.")


def synth_pattern(e_inject, aniso_inject, noise_abs, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:125, 0:125]
    dx, dy = xx-62.5, yy-62.5
    rr = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    img = 8.0*np.exp(-rr/22.0)
    img *= (1 + (aniso_inject-1)*np.cos(th)**2)   # max/min = aniso_inject
    th0 = np.deg2rad(94.0)
    ct, st = np.cos(th0), np.sin(th0)
    centers = [(62.5, 62.5)]                      # beam first, beam_idx=0
    for r0, nd, amp in ((30, 10, 2500.0), (48, 14, 1200.0)):
        for k in range(nd):
            a = 2*np.pi*k/nd
            r = r0*(1 + e_inject*np.cos(2*(a-th0)))
            xc, yc = 62.5 + r*np.cos(a), 62.5 + r*np.sin(a)
            centers.append((yc, xc))
            ddx, ddy = xx-xc, yy-yc
            u = (ddx*ct + ddy*st)/(4.0*(1+e_inject))
            v = (-ddx*st + ddy*ct)/(4.0*(1-e_inject))
            img += amp*np.exp(-(u*u + v*v)/2)
    d2 = (xx-62.5)**2 + (yy-62.5)**2
    img += 20000*np.exp(-d2/(2*5.0**2))
    if noise_abs > 0:
        img += rng.normal(0, noise_abs, img.shape)
    return np.clip(img, 0, None).astype(np.float32), centers


sc1, sc2 = st.columns([1, 2])
with sc1:
    e_inj = st.slider("Injected elliptical distortion", 0.0, 0.15, 0.07,
                      0.01,
                      help="0.07 reproduces the real dataset's distortion.")
    a_inj = st.slider("Injected diffuse anisotropy", 1.0, 3.0, 1.0, 0.1,
                      help="1.0 = perfectly isotropic input. The median "
                           "background subtraction mostly removes smooth "
                           "large-scale structure — see note below.")
    noise_inj = st.slider("Noise (counts sigma)", 0.0, 30.0, 5.0, 1.0)
    med_syn = st.slider("Median background size (px)", 5, 41, 21, 2,
                        key="med_syn")
    seed_inj = st.slider("Random seed", 0, 9, 0, 1)

with sc2:
    simg, scenter = synth_pattern(e_inj, a_inj, noise_inj, seed_inj)
    # circular masks (the naive instrument)
    mc = build_mask(simg.shape, np.array(scenter), 0)
    tc = isolate_residual(simg, mc, median_size=med_syn)
    ac, _ = anisotropy(tc, mc)
    # elliptical masks (the corrected instrument), e known = injected
    me = build_elliptical_mask(simg.shape, np.array(scenter), 0,
                               e_inj, np.deg2rad(94.0))
    te = isolate_residual(simg, me, median_size=med_syn)
    ae, _ = anisotropy_elliptical(te, me, 62.5, 62.5, e_inj,
                                  np.deg2rad(94.0))
    r1, r2, r3 = st.columns(3)
    r1.metric("Injected anisotropy", f"{a_inj:.2f}")
    r2.metric("Recovered (circular masks)", f"{ac:.2f}")
    r3.metric("Recovered (elliptical masks)", f"{ae:.2f}")
    if abs(a_inj-1.0) < 1e-9 and ac > 1.2 and ae < 1.2:
        st.success("The confound, live: perfectly isotropic input, yet "
                   "circular masks report anisotropy — and the elliptical "
                   "correction removes it.")
    elif ae <= ac:
        st.info("Elliptical masks recover closer to the injected value.")
    else:
        st.warning("Unexpected: circular masks did better here — worth "
                   "investigating, not hiding.")
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.5))
    ax2[0].imshow(simg, norm=LogNorm(vmin=max(simg[simg > 0].min(), 1e-9),
                                     vmax=simg.max()),
                  cmap="inferno", origin="lower")
    ax2[0].set_title(f"synthetic pattern (log), e={e_inj:.2f}")
    ax2[1].imshow(np.log1p(tc), cmap="magma", origin="lower")
    ax2[1].set_title(f"residual, circular masks (log1p), aniso={ac:.2f}")
    for a_ in ax2:
        a_.set_xticks([])
        a_.set_yticks([])
    fig2.tight_layout()
    st.pyplot(fig2)
    st.caption("Note: the 21px median subtraction removes smooth large-scale "
               "structure, so injected *smooth* anisotropy is mostly "
               "subtracted away — the instrument sees what survives the "
               "filter. Shrink the median size to watch smooth structure "
               "reappear in the residual.")

st.markdown("---")
st.header("The full story, in layers")
st.caption("One-minute version first, then the method, then every way the "
           "instrument tried to fool us, then what we still don't know.")

st.subheader("In one minute")
st.markdown(
    "1. A 4D-STEM scan records a diffraction pattern at every probe position "
    "(here 100×84 = 8,400 patterns of 125×125 px).\n"
    "2. Average them into one pattern — the 'splat-map'.\n"
    "3. Find the 33 Bragg disks by cross-correlating with the probe's own "
    "template; black them out (24.3% of pixels) — delete the atoms.\n"
    "4. Inpaint the holes, subtract the smooth background (21px median) → "
    "the diffuse residual: 2.10% of counts.\n"
    "5. Measure its azimuthal structure: 8 wedges in an annulus → max/min "
    "= 1.79, or **2.00–2.06** after correcting the elliptical distortion."
)

with st.expander("The method (technical)"):
    st.markdown(
        "- **Detection:** max-projection cross-correlated with the file's "
        "probe template; local maxima above 1.2% of the correlation maximum "
        "→ 33 centers (clean valley: 33 above, background specks below).\n"
        "- **Mask:** direct beam R=8px (from measured radial profile), "
        "32 Bragg disks R=6px (probe half-max 5px + 1px margin).\n"
        "- **Residual:** masked sites inpainted with the unmasked median; "
        "21px median filter estimates the smooth background; positive "
        "remainder kept; Bragg sites forced to zero.\n"
        "- **Metric:** annulus 19–56px (0.15–0.45 of frame), 8 azimuthal "
        "wedges, anisotropy = max/min wedge mean.\n"
        "- **Elliptical correction:** joint geometric least-squares fit "
        "r(θ)=r₀(1+e·cos2(θ−θ₀)) across all rings → e=0.0729, θ=94.3°, "
        "center (62.71, 63.12); elliptical masks + elliptical annulus "
        "(no image resampling), cross-checked against a dewarped image."
    )

st.subheader("Five ways it tried to fool us")
st.markdown(
    "**1. Elliptical distortion — caught, corrected, falsified as the sole "
    "cause.** The test data deliberately includes ~7% elliptical distortion "
    "(documented). We measured it (joint fit: e=0.0729), corrected it two "
    "independent ways — elliptical masks with no resampling, and a dewarped "
    "image — and both agree: anisotropy 1.79 → 2.00 / 2.06. The distortion "
    "was diluting the signal, not creating it.")
with st.expander("Ablation table"):
    st.markdown(
        "| setup | anisotropy |\n"
        "|---|---|\n"
        "| circular masks + circular annulus (baseline) | 1.79 |\n"
        "| elliptical masks + circular annulus | 2.01 |\n"
        "| circular masks + elliptical annulus | 1.74 |\n"
        "| elliptical masks + elliptical annulus (proper) | **2.00** |\n"
        "| dewarped image, standard pipeline | **2.06** |\n\n"
        "The mask shape carried the correction; the annulus geometry barely "
        "mattered. Dewarp verification: residual ellipticity 0.073 → 0.018.")
st.markdown(
    "**2. Mask-edge sensitivity — characterized, not hidden.** ~98% of the "
    "residual lives within 8px of a mask edge; excluding pixels within 3px "
    "of any edge moves the anisotropy 1.79 → 2.82. The headline number "
    "depends on analysis choices — the live lab above lets you move them.")
st.markdown(
    "**3. Sampling wobble — characterized.** Split the probe positions in "
    "half: 1.84 vs 2.09. A 2.0 alert gate would fire on one half and stay "
    "silent on the other — of the same data. Any screening loop needs "
    "margin and multi-dataset statistics, not a hard gate.")
st.markdown(
    "**4. The phonon verdict — the big one.** `Au_sim` is a calibration "
    "phantom. Its own documentation (Zenodo 10.5281/zenodo.3592520) describes "
    "geometric calibration tests and says nothing about phonons, temperature, "
    "or thermal scattering; the py4DSTEM paper never mentions frozen phonon. "
    "It very likely contains no TDS at all. So 'TDS' on this page is an "
    "*unvalidated proxy label* for 'diffuse background after Bragg "
    "removal'. The instrument is real and reproducible; the physical "
    "interpretation is pending.")
st.markdown(
    "**5. The vacuum null test — run, and it drew the line.** We ran the "
    "unmodified pipeline on the probe-over-vacuum stack from the "
    "full-resolution Zenodo record (true signal: zero). Result: the "
    "instrument reports **1.07% 'residual' from nothing** — a method floor "
    "from beam skirt and the positive-clip bias — but anisotropy of only "
    "**1.06**, essentially isotropic. So the pipeline does *not* hallucinate "
    "azimuthal structure, while the absolute 2.10% figure carries about a "
    "1% floor. Honest reading of Au_sim: ~1% excess residual with real ~2× "
    "azimuthal structure.")

with st.expander("Run log (11 runs — the honest sausage-making)"):
    st.markdown(
        "1. py4DSTEM's Drive downloader broke (gdown API) → downloaded "
        "directly (525.7 MB).\n"
        "2. py4DSTEM 0.14.18 couldn't read the legacy v0.11 file → raw "
        "uint16 via h5py.\n"
        "3. Percentile-threshold mask bug → 100% masked. Fixed.\n"
        "4. p98 threshold → 3 regions; unmasked-Bragg leakage dominated.\n"
        "5–6. p95 → 11 regions, 14.5% masked; faint outer disks missed.\n"
        "7. Probe-template cross-correlation, bad threshold → 1 center.\n"
        "8. Threshold 1.2% of corr max → 33 centers.\n"
        "9. R=5 → direct-beam skirt leaked into residual.\n"
        "10. Flood-fill per disk → bled to 40.8% masked.\n"
        "11. Final: beam R=8 / disks R=6 from measured radial profiles.")

st.subheader("What we don't know")
st.markdown(
    "- What the residual physically *is* in this simulation — "
    "nanoparticle shape-transform fringes? method artifact? a mix?\n"
    "- Whether the anisotropy survives a better background estimator "
    "(radial profile instead of global-median inpaint).\n"
    "- Whether any of this appears in experimental data — the only place "
    "real TDS exists. (Diffraction shifting, also deliberate in this "
    "dataset, is still uncorrected.)\n"
    "- The sharpest next test: the vacuum probe stack (true zero) from "
    "the full Zenodo record — separates method artifact from simulation "
    "content in one run."
)

st.subheader("What's next")
st.markdown(
    "1. **Vacuum-stack null test** — done (see confound #5): ~1% method "
    "floor confirmed; anisotropy metric validated (1.06 on true zero).\n"
    "2. **Better background estimator**, then re-run.\n"
    "3. **Experimental data** — public 4D-STEM datasets where TDS actually "
    "exists.\n"
    "4. **The screening loop** — if the null test passes: sweep open "
    "datasets, flag anomalous anisotropy, alert for review."
)

st.markdown("---")
st.subheader("Code and data")
with st.expander("delete_the_atoms.py (the original batch script)"):
    st.code((HERE / "delete_the_atoms.py").read_text(), language="python")
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("tds_residual.png",
                       data=(HERE / "tds_residual.png").read_bytes(),
                       file_name="tds_residual.png", mime="image/png")
with d2:
    st.download_button("delete_the_atoms.py",
                       data=(HERE / "delete_the_atoms.py").read_bytes(),
                       file_name="delete_the_atoms.py",
                       mime="text/x-python")
with d3:
    st.download_button("tds_core.py",
                       data=(HERE / "tds_core.py").read_bytes(),
                       file_name="tds_core.py", mime="text/x-python")

st.caption(
    "Numbers on this page are recomputed live from the data unless labeled "
    "precomputed. The elliptical-correction constants (e=0.0729, θ=94.3°) "
    "are deterministic fits documented above. Test data: py4DSTEM Au_sim "
    "tutorial dataset (multislice simulation, Zenodo 10.5281/zenodo.3592520)."
)
