# 'Delete the Atoms' — an honest instrument

An interactive instrument for interrogating what remains when you black out
the Bragg peaks in 4D-STEM data — presented honestly, confounds included.

**The one-minute version.** Average a 4D-STEM scan into one diffraction
pattern. Find every Bragg disk. Black them out — delete the atoms. What
remains is the diffuse background: the whispers between the shouts. This
app measures it, then shows you every way the measurement tried to fool
us, with live knobs so you can check the work yourself.

**The honest numbers.** On py4DSTEM's `Au_sim` tutorial dataset (a
calibration phantom — very likely no thermal scattering at all): 2.10% of
counts survive Bragg removal; azimuthal anisotropy 1.79, or 2.00–2.06 after
correcting the dataset's deliberate 7.3% elliptical distortion. A
vacuum-stack null test measured a ~1% method floor; the anisotropy metric
reads 1.06 on true zero. So: ~1% excess residual with real ~2× azimuthal
structure — of unknown physical origin. "TDS" here is an unvalidated proxy
label, stated as such on the page.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud)

Public repo → [share.streamlit.io](https://share.streamlit.io) → New app →
select repo, branch `main`, main file `streamlit_app.py` → Deploy.
No secrets or config needed. Free tier (1 GB RAM) is plenty — the data
arrays are 125×125.

## Files

- `streamlit_app.py` — the app (live lab, synthetic playground, full story)
- `tds_core.py` — the pipeline, single source of truth
- `elliptical_correction.py` — joint ellipse fit, elliptical masks/annulus, dewarp check
- `vacuum_null_test.py` — the null test (reads the Zenodo record over HTTP)
- `delete_the_atoms.py` — the original batch script
- `assets/` — precomputed mean/max diffraction patterns + probe template (.npy)

## What's next

Vacuum null done; remaining: a better background estimator, validation on
experimental 4D-STEM data where real TDS exists, then the screening loop
(sweep open datasets, flag anomalous anisotropy).
