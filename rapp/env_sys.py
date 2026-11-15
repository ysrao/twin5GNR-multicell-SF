"""
System-level environment for the slice-optimiser rApp, built on
NVIDIA Sionna SYS (sionna.sys), following the "Sionna SYS meets
Sionna RT" pattern: channels come from a geometry model, while
scheduling support, link adaptation and PHY abstraction — the parts
that turn SINR into delivered bits — are Sionna SYS components.

What Sionna SYS computes here (per candidate scheduling profile):
  * PHYAbstraction           — SINR -> BLER -> decoded bits (EESM tables);
    the link-adaptation table (SINR -> MCS) is itself calibrated from
    PHYAbstraction at environment initialisation
  * proportional-fair allocation across the profile's slice PRB shares

Geometry (this package's reference cell, matching the SF twin work):
  one tower, three sectors at 90/210/330 deg, 30 m AGL, 3.6 GHz,
  TR 38.901 UMi pathloss, TR 38.901 sector pattern, per-RE noise
  -122.2 dBm at 30 kHz SCS and 7 dB noise figure.

KPIs returned: median DL, edge DL (5th pct), fairness (Jain), PRB
utilisation — computed from SYS outputs. Uplink and signalling KPIs
are NOT produced by this downlink environment; the reference-scoring
assumptions for them are documented in README.md and marked in code
(UL follows the DL response at the target ratio 0.3; signalling
improves with PRB headroom). In deployment, all seven KPIs come from
measured PM counters and SYS is used to evaluate candidate profiles
before actuation — the same train-slow / act-fast split.
"""

import numpy as np
import torch

from sionna.sys import PHYAbstraction

from policy import PROFILES

FC_GHZ = 3.6
H_BS, H_UT = 30.0, 1.5
EIRP_DBM = 47.0
N_PRB = 273                      # 100 MHz, 30 kHz SCS
RE_PER_PRB_SLOT = 12 * 14        # subcarriers x OFDM symbols
NOISE_PER_RE_DBM = -122.2        # 30 kHz RE, NF 7 dB (as in the twin paper)
SECTORS_DEG = (90.0, 210.0, 330.0)
SLICES = ("eMBB", "URLLC", "mMTC")


def _umi_pathloss_db(d2d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """TR 38.901 UMi street-canyon pathloss with probabilistic LOS."""
    d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)
    d_bp = 4 * H_BS * H_UT * FC_GHZ * 1e9 / 3e8
    pl_los = np.where(
        d2d <= d_bp,
        32.4 + 21 * np.log10(d3d) + 20 * np.log10(FC_GHZ),
        32.4 + 40 * np.log10(d3d) + 20 * np.log10(FC_GHZ)
        - 9.5 * np.log10(d_bp ** 2 + (H_BS - H_UT) ** 2))
    pl_nlos = np.maximum(
        pl_los,
        35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(FC_GHZ) - 0.3 * (H_UT - 1.5))
    p_los = np.where(d2d <= 18, 1.0,
                     18 / d2d + np.exp(-d2d / 36) * (1 - 18 / d2d))
    is_los = rng.random(len(d2d)) < p_los
    shadow = np.where(is_los, rng.normal(0, 4.0, len(d2d)),
                      rng.normal(0, 7.8, len(d2d)))
    return np.where(is_los, pl_los, pl_nlos) + shadow


def _sector_gain_db(delta_deg: np.ndarray) -> np.ndarray:
    """TR 38.901 sector pattern, 65 deg HPBW, 30 dB front-to-back."""
    return -np.minimum(12 * (delta_deg / 65.0) ** 2, 30.0)


class SysCellEnv:
    """One tower, three sectors, three slices. Evaluates a scheduling
    profile with Sionna SYS and returns the KPI vector."""

    def __init__(self, n_users: int = 30, n_slots: int = 150, seed: int = 0):
        self.n_users = n_users
        self.n_slots = n_slots
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        # User drop: uniform in a 250 m disc, assigned to nearest sector.
        ang = rng.uniform(0, 2 * np.pi, n_users)
        rad = 250 * np.sqrt(rng.uniform(0.05, 1.0, n_users))
        self.user_xy = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        self.user_d = np.maximum(rad, 10.0)
        self.user_az = np.degrees(np.arctan2(self.user_xy[:, 1], self.user_xy[:, 0])) % 360
        self.user_sector = np.argmin(
            np.abs((self.user_az[:, None] - np.array(SECTORS_DEG)[None, :] + 180) % 360 - 180),
            axis=1)
        self.user_slice = rng.integers(0, 3, n_users)  # slice membership
        self.pathloss = _umi_pathloss_db(self.user_d, rng)
        # Per-sector antenna gain toward each user (desired + interferers)
        self.gain = np.stack([
            _sector_gain_db((self.user_az - az + 180) % 360 - 180)
            for az in SECTORS_DEG], axis=1)  # [user, sector]
        self.phy_abs = PHYAbstraction()
        self.mcs_thresholds = self._calibrate_mcs_thresholds()

    def _calibrate_mcs_thresholds(self) -> np.ndarray:
        """Link-adaptation table calibrated from Sionna SYS itself:
        for each MCS index, the lowest SINR (dB) at which PHYAbstraction
        reports a successful decode. One batched SYS call."""
        sinr_grid = np.arange(-10.0, 35.5, 1.0, dtype=np.float32)
        mcs_rows, sinr_rows = [], []
        for m in range(28):
            mcs_rows.append(np.full(len(sinr_grid), m, dtype=np.int32))
            sinr_rows.append(sinr_grid)
        mcs_t = torch.tensor(np.concatenate(mcs_rows).reshape(1, -1, 1))
        sinr_t = torch.tensor(np.concatenate(sinr_rows).reshape(1, -1, 1))
        re_t = torch.full((1, len(mcs_t.flatten()), 1), 2000.0)
        out = self.phy_abs(mcs_t, sinr_eff=sinr_t, num_allocated_re=re_t)
        success = out[1].detach().cpu().numpy().flatten().reshape(28, len(sinr_grid))
        thresholds = np.full(28, 99.0)
        for m in range(28):
            ok = np.where(success[m] > 0)[0]
            if len(ok):
                thresholds[m] = sinr_grid[ok[0]]
        return thresholds

    def _pick_mcs(self, sinr_db: np.ndarray) -> np.ndarray:
        mcs = np.searchsorted(self.mcs_thresholds, sinr_db) - 1
        return np.clip(mcs, 0, 27)

    def _sinr_per_re(self, fading: np.ndarray) -> np.ndarray:
        """Per-RE SINR (linear) for every user in one slot."""
        p_tx_re = EIRP_DBM - 8.0 - 10 * np.log10(N_PRB * 12)  # conducted per RE
        rx = p_tx_re + self.gain - self.pathloss[:, None]     # [user, sector] dBm
        rx_lin = 10 ** (rx / 10) * fading                     # fading per link
        sig = rx_lin[np.arange(self.n_users), self.user_sector]
        interf = rx_lin.sum(axis=1) - sig
        noise = 10 ** (NOISE_PER_RE_DBM / 10)
        return sig / (interf + noise)

    def _window_kpis(self, bits_window: np.ndarray, prb_used: float,
                     n_slots: int) -> np.ndarray:
        user_mbps = bits_window / (n_slots * 1e-3) / 1e6
        active = user_mbps[user_mbps > 0]
        if len(active) == 0:
            return np.array([0, 0, 0, 0, 0, 100.0, 60.0])
        median_dl = float(np.median(active))
        edge_dl = float(np.percentile(active, 5))
        fairness = float((active.sum() ** 2) / (len(active) * (active ** 2).sum()))
        prb_util = float(100.0 * prb_used / (n_slots * N_PRB))
        # Reference-scoring assumptions (documented, NOT SYS outputs):
        median_ul, edge_ul = 0.3 * median_dl, 0.3 * edge_dl
        signalling = float(np.clip(55.0 - 0.35 * prb_util, 12.0, 60.0))
        return np.array([median_dl, median_ul, edge_dl, edge_ul,
                         fairness, prb_util, signalling])

    def simulate(self, choose_action, n_windows: int = 6,
                 window_slots: int = 25, load: float = 0.75):
        """Run the SYS slot loop in decision windows — the rApp cadence.
        choose_action(kpis_so_far) -> profile index, called at each window
        boundary. Returns a list of (kpis_before, action, kpis_after)."""
        rng = self.rng
        n = self.n_users
        avg_rate = np.full(n, 1e-9)
        # Warm-up window under the balanced profile to seed the state
        trajectory = []
        kpis = self._run_window(0, window_slots, load, avg_rate)
        for _ in range(n_windows):
            action = int(choose_action(kpis))
            new_kpis = self._run_window(action, window_slots, load, avg_rate)
            trajectory.append((kpis, action, new_kpis))
            kpis = new_kpis
        return trajectory

    def _run_window(self, profile_idx: int, n_slots: int, load: float,
                    avg_rate: np.ndarray) -> np.ndarray:
        _, shares, _ = PROFILES[profile_idx]
        rng = self.rng
        n = self.n_users
        bits_window = np.zeros(n)
        prb_used = 0.0
        for _ in range(n_slots):
            fading = rng.exponential(1.0, size=(n, 3))
            sinr = self._sinr_per_re(fading)
            sinr_db = 10 * np.log10(np.maximum(sinr, 1e-9))
            inst_se = np.log2(1 + sinr)
            pf_metric = inst_se / avg_rate
            active_user = rng.random(n) < load  # finite offered load
            alloc_re = np.zeros(n)
            for s in range(3):
                members = np.where((self.user_slice == s) & active_user)[0]
                if len(members) == 0:
                    continue
                budget_re = shares[s] * N_PRB * RE_PER_PRB_SLOT
                pick = members[np.argmax(pf_metric[members])]
                alloc_re[pick] += budget_re
            scheduled = alloc_re > 0
            if not scheduled.any():
                continue
            prb_used += alloc_re.sum() / RE_PER_PRB_SLOT
            # Sionna SYS PHY abstraction (batched, [batch=1, num_ut, 1])
            idx = np.where(scheduled)[0]
            mcs_t = torch.tensor(self._pick_mcs(sinr_db[idx]).reshape(1, -1, 1),
                                 dtype=torch.int32)
            sinr_t = torch.tensor(sinr_db[idx].reshape(1, -1, 1), dtype=torch.float32)
            re_t = torch.tensor(alloc_re[idx].reshape(1, -1, 1), dtype=torch.float32)
            out = self.phy_abs(mcs_t, sinr_eff=sinr_t, num_allocated_re=re_t)
            decoded = out[0].detach().cpu().numpy().flatten().astype(float)
            bits_window[idx] += decoded
            slot_rate = np.zeros(n)
            slot_rate[idx] = decoded / 1e-3  # bits per 1 ms slot -> bit/s
            avg_rate[:] = 0.98 * avg_rate + 0.02 * (slot_rate / 1e6 + 1e-9)
        return self._window_kpis(bits_window, prb_used, n_slots)

    def evaluate(self, profile_idx: int) -> np.ndarray:
        """KPI vector for one fixed profile over the full horizon."""
        traj = self.simulate(lambda k: profile_idx, n_windows=5,
                             window_slots=self.n_slots // 5)
        return traj[-1][2]
