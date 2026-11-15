"""Multi-site (cluster) extension of the twin optimiser.

The single-site scorer treats neighbour cells as a calibrated
margin. Here the twin holds one isotropic trace per site over a
SHARED street receiver set, so inter-site interference is computed,
not margined: at every receiver all nine sectors (3 sites x 3
sectors) are scored jointly, the serving sector is the strongest,
and SINR is against the other eight sectors plus per-RE noise.

The guarded search is coordinate ascent ACROSS sites: optimise one
site's azimuths/tilts/power holding the others fixed, move to the
next site, repeat until no site improves. Guardrails are evaluated
on the cluster KPIs (coverage, cell-edge) against the as-built
cluster, exactly as the single-site gate works on site KPIs.

Per-site sector RSRP blocks are cached by configuration, so a
candidate that varies one site recomputes only that site's block.
"""
import copy

import numpy as np

import twin as twin_mod
from search import TILTS, AZ_OFFS, POWERS, passes_guardrails


class ClusterScorer:
    """Joint scoring of a set of sites over a shared receiver set."""

    def __init__(self, traces, sites, thresholds=None, bits_per_re=None):
        """traces: {site_id: trace dict from twin.trace_tower} -- all
        traces must share the same receiver array. sites: the site
        inventory dicts (position, as-built config)."""
        self.site_ids = [s["site_id"] for s in sites]
        self.sites = {s["site_id"]: s for s in sites}
        self.rx = traces[self.site_ids[0]]["rx"]
        for sid in self.site_ids:
            assert np.allclose(traces[sid]["rx"], self.rx), \
                f"trace for {sid} uses a different receiver set"
        if thresholds is None:
            thresholds, bits_per_re = twin_mod.calibrate_link_curve()
        self.thresholds, self.bits_per_re = thresholds, bits_per_re
        # one TwinScorer per site supplies the pattern machinery
        self._scorers = {
            sid: twin_mod.TwinScorer(traces[sid],
                                     self.sites[sid]["position_enu_m"],
                                     thresholds, bits_per_re)
            for sid in self.site_ids
        }
        self._block_cache = {}

    def sector_block(self, site_id, config):
        """[rx, 3] RSRP dBm per RE of one site under a config."""
        key = (site_id, tuple(config["azimuths"]), tuple(config["tilts"]),
               float(config.get("power_offset_db", 0.0)))
        if key not in self._block_cache:
            self._block_cache[key] = self._scorers[site_id].sector_rsrp(config)
        return self._block_cache[key]

    def joint_rsrp(self, configs):
        """[rx, 9] RSRP: columns ordered by site, then sector."""
        return np.concatenate(
            [self.sector_block(sid, configs[sid]) for sid in self.site_ids],
            axis=1)

    def score(self, configs, weights):
        """Cluster KPIs for a joint config {site_id: config} and a
        per-receiver traffic weight vector."""
        R = self.joint_rsrp(configs)
        flat = np.where(np.isnan(R), -np.inf, R)
        best = flat.max(axis=1)
        have = np.isfinite(best)
        serving = flat.argmax(axis=1)
        p_lin = 10 ** (flat / 10.0)
        p_best = p_lin[np.arange(len(R)), serving]
        p_int = p_lin.sum(axis=1) - p_best
        noise = 10 ** (twin_mod.NOISE_PER_RE_DBM / 10.0)
        sinr = np.where(have, 10 * np.log10(np.maximum(
            p_best, 1e-300) / (p_int + noise)), np.nan)
        se = np.where(have, self._scorers[self.site_ids[0]].se_of_sinr(
            np.nan_to_num(sinr)), 0.0)
        w = weights * have
        per_site_cov = {}
        for i, sid in enumerate(self.site_ids):
            m = have & (serving // 3 == i)
            per_site_cov[sid] = (
                float(100 * np.mean(best[m] >= twin_mod.COVERAGE_THRESHOLD))
                if m.any() else None)
        return {
            "path_pct": float(100 * have.mean()),
            "median_rsrp": float(np.median(best[have])),
            "edge_rsrp_p5": float(np.percentile(best[have], 5)),
            "coverage_pct": float(
                100 * np.mean(best[have] >= twin_mod.COVERAGE_THRESHOLD)),
            "median_sinr": float(np.median(sinr[have])),
            "capacity_weighted_se": float((w * se).sum() / max(w.sum(), 1e-9)),
            "served_index": float((w * se).sum()),
            "coverage_by_serving_site": per_site_cov,
            "serving_share_by_site": [
                float(np.mean((serving[have] // 3) == i))
                for i in range(len(self.site_ids))],
        }

    def traffic_weights(self, base_configs, site_totals, sector_shares):
        """Per-receiver weights under a traffic scenario: each
        receiver counts for the (site total x sector share) of the
        sector that serves it under the base joint configuration."""
        R = self.joint_rsrp(base_configs)
        serving = np.where(np.isnan(R), -np.inf, R).argmax(axis=1)
        w = np.zeros(len(R))
        for i, sid in enumerate(self.site_ids):
            for s in range(3):
                m = serving == 3 * i + s
                w[m] = site_totals[sid] * sector_shares[sid][s]
        return w


def optimise_cluster(scorer, base_configs, w_projected, w_current, policy,
                     max_rounds=3):
    """Coordinate ascent across sites. Returns (best_configs,
    base_kpis_proj, best_kpis_proj, best_kpis_current, history).
    Guardrails and objective are the cluster KPIs under projected
    traffic; current-traffic KPIs are reported alongside."""
    base_proj = scorer.score(base_configs, w_projected)
    best = copy.deepcopy(base_configs)
    best_proj = base_proj
    history = []

    def consider(site_id, cfg_site, label):
        nonlocal best, best_proj
        trial = copy.deepcopy(best)
        trial[site_id] = cfg_site
        k = scorer.score(trial, w_projected)
        ok = passes_guardrails(k, base_proj, policy)
        history.append({"site": site_id, "label": label,
                        "capacity": k["capacity_weighted_se"],
                        "coverage": k["coverage_pct"],
                        "guardrail_pass": bool(ok)})
        if ok and k["capacity_weighted_se"] > \
                best_proj["capacity_weighted_se"] + 1e-9:
            best, best_proj = trial, k
            return True
        return False

    for _ in range(max_rounds):
        improved = False
        for sid in scorer.site_ids:
            base_site = base_configs[sid]
            for s in range(3):
                for t in TILTS:
                    if t == best[sid]["tilts"][s]:
                        continue
                    cfg = copy.deepcopy(best[sid])
                    cfg["tilts"][s] = t
                    improved |= consider(sid, cfg, f"tilt sector{s} -> {t}")
                for off in AZ_OFFS:
                    cur = best[sid]["azimuths"][s] - base_site["azimuths"][s]
                    if off == cur:
                        continue
                    cfg = copy.deepcopy(best[sid])
                    cfg["azimuths"][s] = base_site["azimuths"][s] + off
                    improved |= consider(sid, cfg,
                                         f"azimuth sector{s} -> {off:+}")
            for p in POWERS:
                if p == best[sid].get("power_offset_db", 0.0):
                    continue
                cfg = copy.deepcopy(best[sid])
                cfg["power_offset_db"] = p
                improved |= consider(sid, cfg, f"power -> {p:+} dB")
        if not improved:
            break
    best_cur = scorer.score(best, w_current)
    return best, base_proj, best_proj, best_cur, history
