"""Sionna RT twin of a single tower on the open SF scene, with
Sionna SYS capacity scoring.

Method (the pattern of the SF twin paper, Rao et al. 2026): ONE
isotropic ray trace per tower; every candidate configuration
(per-sector azimuth/downtilt, site power offset) is then scored in
post-processing by applying the TR 38.901 element pattern
(65 deg horizontal and vertical beamwidths, 30 dB caps, 8 dBi element
gain) plus the 4x4 array factor (12.04 dB) per departure direction --
exactly the accounting of the original rt_multicell.py
(EIRP 47 dBm at boresight = 26.96 conducted + 8 + 12.04).
Stored RSRP is per resource element (n_re = 3264 at 100 MHz /
30 kHz SCS; -35.14 dB), matching the paper.

Capacity: per-receiver SINR (best sector vs the site's other two
sectors + per-RE noise -122.2 dBm at NF 7) is mapped to spectral
efficiency through a link curve calibrated from Sionna SYS
PHYAbstraction. Traffic scenarios (current / projected) weight the
receivers by sector traffic share.
"""
import json
import os
import time

import numpy as np

P_COND_DBM = 26.96          # conducted power (EIRP 47 at boresight)
PER_RE_DB = 35.14           # 10*log10(3264)
NOISE_PER_RE_DBM = -122.2   # NF 7 dB
ELEM_GAIN_DBI = 8.0
ARRAY_GAIN_DB = 12.04       # 4x4
COVERAGE_THRESHOLD = -90.0  # dBm per RE (paper's good-coverage bar)


def pattern_db(phi_deg, theta_deg, az_deg, tilt_deg):
    """TR 38.901 element pattern + array factor, dB, per departure dir."""
    dphi = (phi_deg - az_deg + 180.0) % 360.0 - 180.0
    ah = -np.minimum(12.0 * (dphi / 65.0) ** 2, 30.0)
    dv = (90.0 - theta_deg) + tilt_deg   # path elevation rel. to boresight
    av = -np.minimum(12.0 * (dv / 65.0) ** 2, 30.0)
    return np.maximum(ah + av, -30.0) + ELEM_GAIN_DBI + ARRAY_GAIN_DB


def street_receivers(ply_path, cx, cy, half=150.0, step=20.0):
    """Street-level receivers around (cx, cy): surface from the scene
    mesh, local ground = 5th percentile in a 60 m window, keep points
    whose surface is < 2.5 m above local ground (the notebook rule)."""
    import trimesh
    mesh = trimesh.load(ply_path)
    xs = np.arange(cx - half, cx + half + 1, step)
    ys = np.arange(cy - half, cy + half + 1, step)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
    origins = np.concatenate([grid, np.full((len(grid), 1), 60.0)], axis=1)
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (len(grid), 1))
    locs, idx_ray, _ = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=False)
    surf = {}
    for loc, ri in zip(locs, idx_ray):
        key = (grid[ri][0], grid[ri][1])
        if key not in surf or loc[2] > surf[key]:
            surf[key] = loc[2]
    gx, gy = grid[:, 0], grid[:, 1]
    sz = np.array([surf.get((x, y), np.nan) for x, y in grid])
    pts = []
    for i in range(len(grid)):
        m = (np.abs(gx - gx[i]) <= 60) & (np.abs(gy - gy[i]) <= 60) \
            & ~np.isnan(sz)
        if not m.any() or np.isnan(sz[i]):
            continue
        ground = np.percentile(sz[m], 5)
        if sz[i] - ground < 2.5:
            pts.append((float(gx[i]), float(gy[i]), float(ground + 1.5)))
    return np.array(pts)


def trace_tower(ply_path, tx_pos, rx_pts, samples=500_000, seed=41,
                scene_xml="/tmp/rapp_twin_scene.xml"):
    """One isotropic trace; returns per-receiver path gains and
    departure angles (the reusable twin state)."""
    import mitsuba as mi
    if mi.variant() is None:
        # Only choose a variant when none is active, and prefer the GPU
        # variant on GPU hosts: forcing LLVM on a CUDA machine leaves
        # Sionna RT's itu-radio-material plugin unregistered there.
        try:
            import torch
            mi.set_variant("cuda_ad_mono_polarized"
                           if torch.cuda.is_available()
                           else "llvm_ad_mono_polarized")
        except Exception:
            pass
    from sionna.rt import (load_scene, Transmitter, Receiver,
                           PlanarArray, PathSolver)
    with open(scene_xml, "w") as f:
        f.write(f'''<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="itu_concrete">
    <string name="type" value="concrete"/>
    <float name="thickness" value="0.1"/>
    <float name="scattering_coefficient" value="0.3"/>
  </bsdf>
  <shape type="ply" id="city">
    <string name="filename" value="{os.path.abspath(ply_path)}"/>
    <ref id="itu_concrete" name="bsdf"/>
  </shape>
</scene>''')
    scene = load_scene(scene_xml)
    scene.frequency = 3.6e9
    if "itu_concrete" in scene.radio_materials:
        scene.radio_materials["itu_concrete"].scattering_coefficient = 0.3
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.add(Transmitter(name="tx", position=[float(v) for v in tx_pos]))
    for i, p in enumerate(rx_pts):
        scene.add(Receiver(name=f"rx{i}",
                           position=[float(p[0]), float(p[1]), float(p[2])]))
    t0 = time.time()
    paths = PathSolver()(scene, max_depth=3, samples_per_src=samples,
                         synthetic_array=False, los=True,
                         specular_reflection=True, diffuse_reflection=True,
                         diffraction=True, edge_diffraction=True, seed=seed)
    a = np.abs(np.array(paths.a)) ** 2
    if a.ndim == 6:            # leading axis: field components
        a = a.sum(axis=0)
    gain = a[:, 0, 0, 0, :]                       # [rx, paths]
    phi = np.degrees(np.array(paths.phi_t)[:, 0, 0, 0, :])
    theta = np.degrees(np.array(paths.theta_t)[:, 0, 0, 0, :])
    return {"gain": gain, "phi": phi, "theta": theta, "rx": rx_pts,
            "trace_s": time.time() - t0, "samples": samples}


def calibrate_link_curve():
    """Spectral efficiency vs SINR from Sionna SYS PHYAbstraction:
    per MCS, the lowest SINR with a successful decode at 2000 RE, and
    the decoded bits per RE at high SINR."""
    import torch
    from sionna.sys import PHYAbstraction
    phy = PHYAbstraction()
    n_re = 2000
    sinr_grid = np.arange(-10.0, 36.0, 1.0)
    mcs_rows, sinr_rows = [], []
    for m in range(28):
        for s in sinr_grid:
            mcs_rows.append(m)
            sinr_rows.append(s)
    mcs_t = torch.tensor(np.array(mcs_rows).reshape(1, -1, 1),
                         dtype=torch.int32)
    sinr_t = torch.tensor(np.array(sinr_rows).reshape(1, -1, 1),
                          dtype=torch.float32)
    re_t = torch.full((1, len(mcs_rows), 1), float(n_re))
    out = phy(mcs_t, sinr_eff=sinr_t, num_allocated_re=re_t)
    bits = np.array(out[0]).flatten()
    success = np.array(out[1]).flatten() > 0
    bits = bits.reshape(28, len(sinr_grid))
    success = success.reshape(28, len(sinr_grid))
    thresholds = np.full(28, np.inf)
    bits_per_re = np.zeros(28)
    for m in range(28):
        ok = np.where(success[m])[0]
        if len(ok):
            thresholds[m] = sinr_grid[ok[0]]
            bits_per_re[m] = bits[m, -1] / n_re
    return thresholds, bits_per_re


class TwinScorer:
    """Scores candidate configurations against a completed trace."""

    def __init__(self, trace, tx_xy, thresholds=None, bits_per_re=None):
        self.gain, self.phi, self.theta = (trace["gain"], trace["phi"],
                                           trace["theta"])
        self.rx = trace["rx"]
        d = self.rx[:, :2] - np.array(tx_xy)
        self.dist = np.hypot(d[:, 0], d[:, 1])
        self.az_of_rx = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 360) % 360
        if thresholds is None:
            thresholds, bits_per_re = calibrate_link_curve()
        self.thresholds, self.bits_per_re = thresholds, bits_per_re

    def sector_rsrp(self, config):
        """[rx, 3] RSRP dBm per RE for a config dict with azimuths,
        tilts (deg) and power_offset_db."""
        out = []
        for az, tilt in zip(config["azimuths"], config["tilts"]):
            g = 10 ** (pattern_db(self.phi, self.theta, az, tilt) / 10.0)
            p = (self.gain * g).sum(axis=1)
            rsrp = np.where(
                p > 0, P_COND_DBM + config.get("power_offset_db", 0.0)
                + 10 * np.log10(np.maximum(p, 1e-300)) - PER_RE_DB, np.nan)
            out.append(rsrp)
        return np.stack(out, axis=1)

    def se_of_sinr(self, sinr_db):
        idx = np.searchsorted(self.thresholds, sinr_db, side="right") - 1
        return np.where(idx >= 0, self.bits_per_re[np.clip(idx, 0, 27)], 0.0)

    def score(self, config, traffic_weights):
        """traffic_weights: [n_rx] scenario weights (from the traffic
        model). Returns the KPI dict used by the guardrails/objective."""
        R = self.sector_rsrp(config)
        best = np.nanmax(np.where(np.isnan(R), -np.inf, R), axis=1)
        have = np.isfinite(best)
        serving = np.nanargmax(np.where(np.isnan(R), -np.inf, R), axis=1)
        p_lin = 10 ** (np.where(np.isnan(R), -np.inf, R) / 10.0)
        p_best = p_lin[np.arange(len(R)), serving]
        p_int = p_lin.sum(axis=1) - p_best
        noise = 10 ** (NOISE_PER_RE_DBM / 10.0)
        sinr = np.where(have, 10 * np.log10(np.maximum(
            p_best, 1e-300) / (p_int + noise)), np.nan)
        se = np.where(have, self.se_of_sinr(np.nan_to_num(sinr)), 0.0)
        w = traffic_weights * have
        return {
            "path_pct": float(100 * have.mean()),
            "median_rsrp": float(np.median(best[have])),
            "edge_rsrp_p5": float(np.percentile(best[have], 5)),
            "coverage_pct": float(100 * np.mean(best[have] >= COVERAGE_THRESHOLD)),
            "median_sinr": float(np.median(sinr[have])),
            "capacity_weighted_se": float((w * se).sum() / max(w.sum(), 1e-9)),
            "served_index": float((w * se).sum()),
            "serving_sector_share": [
                float(np.mean(serving[have] == s)) for s in range(3)],
        }


def traffic_weights(scorer, base_config, sector_traffic):
    """Per-receiver weights: each receiver counts for the traffic of
    the sector that serves it under the base configuration."""
    R = scorer.sector_rsrp(base_config)
    serving = np.nanargmax(np.where(np.isnan(R), -np.inf, R), axis=1)
    return np.array([sector_traffic[s] for s in serving], dtype=float)
