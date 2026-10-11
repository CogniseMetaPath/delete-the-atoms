#!/usr/bin/env python3
"""Vacuum-stack null test for 'Delete the Atoms'.

Runs the UNMODIFIED pipeline on the probe-over-vacuum stack
(4DSTEM_experiment/data/diffractionslices/probe_template/data,
shape (250,250,20)) from the full-resolution Zenodo record, read via
HTTP range requests (no 2.1GB download).

True TDS = 0. Any 'residual' or 'anisotropy' reported here is pure
method artifact.

Interpretation guide:
  tds_frac ~ 0% and anisotropy ~ 1.0  -> method is clean; the Au_sim
      residual's structure comes from the simulation content.
  tds_frac >> 0% or anisotropy >> 1.0 -> the pipeline invents signal
      from nothing; the Au_sim numbers are (at least partly) artifact.
"""
import numpy as np
import fsspec
import h5py
from tds_core import find_disks, build_mask, isolate_residual, anisotropy

URL = ("https://zenodo.org/api/records/3592520/files/"
       "calibrationData_circularProbe.h5/content")
VKEY = "4DSTEM_experiment/data/diffractionslices/probe_template/data"


def main():
    fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
    h5 = h5py.File(fs.open(URL, mode="rb"), "r")
    dset = h5[VKEY]
    print(f"vacuum stack: {VKEY} shape={dset.shape} dtype={dset.dtype}",
          flush=True)
    stack = dset[...].astype(np.float32)   # (250,250,20), ~2.5MB
    if stack.shape[2] == 20:
        stack = np.moveaxis(stack, 2, 0)   # -> (20, 250, 250)
    print(f"stack: {stack.shape}", flush=True)
    mean_dp = stack.mean(axis=0)
    max_dp = stack.max(axis=0)
    print(f"mean pattern: total={mean_dp.sum():.3e}, "
          f"max px={mean_dp.max():.3e}", flush=True)

    # template = the beam itself (mean of the stack)
    probe = mean_dp.astype(np.float32)
    centers, beam_idx, r_probe = find_disks(max_dp, probe)
    print(f"\ndisk centers found: {len(centers)} "
          f"(expect 1: the direct beam)", flush=True)
    # full-res frame is 2x the Drive copy's 125px -> scale mask radii
    mask = build_mask(mean_dp.shape, centers, beam_idx,
                      r_beam=16, r_disk=12)
    tds = isolate_residual(mean_dp, mask)
    tds_frac = 100*tds.sum()/mean_dp.sum()
    aniso, wedges = anisotropy(tds, mask)
    print(f"masked: {100*mask.mean():.2f}% of pixels")
    print(f"NULL RESULT: tds_frac = {tds_frac:.4f}% | "
          f"anisotropy = {aniso:.3f}")
    print(f"wedge means: {np.round(wedges, 4)}")
    print("DONE")


if __name__ == "__main__":
    main()
