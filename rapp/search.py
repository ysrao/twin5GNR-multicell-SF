"""Guarded configuration search over the twin.

Coordinate ascent over per-sector downtilt (0-10 deg), per-sector
azimuth offset (-10..+10 deg) and site power offset (-2..+2 dB).
Objective: capacity under the PROJECTED traffic scenario.
Guardrails (from the A1 policy): coverage must not fall below the
as-built coverage, and the cell-edge (5th percentile) RSRP must not
drop by more than the allowed margin. A candidate that fails the
gate is never emitted -- the rApp falls back to the as-built
configuration, so the twin can only propose changes it has verified
in simulation (the guarded-acceptance pattern).
"""
import copy

TILTS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
AZ_OFFS = [-10.0, -5.0, 0.0, 5.0, 10.0]
POWERS = [-2.0, 0.0, 2.0]


def objective(kpis):
    return kpis["capacity_weighted_se"]


def passes_guardrails(kpis, base, policy):
    return (
        kpis["coverage_pct"] >= base["coverage_pct"]
        + policy.get("min_coverage_delta_pts", 0.0)
        and kpis["edge_rsrp_p5"] >= base["edge_rsrp_p5"]
        - policy.get("max_edge_drop_db", 1.0)
    )


def optimise(scorer, base_config, w_projected, w_current, policy,
             max_passes=4):
    """Returns (best_config, base_kpis_proj, best_kpis_proj,
    best_kpis_current, history). Scoring for the objective and the
    guardrails is on the projected-traffic scenario; current-traffic
    KPIs are reported alongside."""
    base_proj = scorer.score(base_config, w_projected)
    best_cfg = copy.deepcopy(base_config)
    best_proj = base_proj
    history = []

    def consider(cfg, label):
        nonlocal best_cfg, best_proj
        k = scorer.score(cfg, w_projected)
        ok = passes_guardrails(k, base_proj, policy)
        history.append({"label": label, "config": copy.deepcopy(cfg),
                        "capacity": k["capacity_weighted_se"],
                        "coverage": k["coverage_pct"],
                        "guardrail_pass": bool(ok)})
        if ok and objective(k) > objective(best_proj) + 1e-9:
            best_cfg, best_proj = copy.deepcopy(cfg), k
            return True
        return False

    for _ in range(max_passes):
        improved = False
        for s in range(3):
            for t in TILTS:
                if t == best_cfg["tilts"][s]:
                    continue
                cfg = copy.deepcopy(best_cfg)
                cfg["tilts"][s] = t
                improved |= consider(cfg, f"tilt sector{s} -> {t}")
            for off in AZ_OFFS:
                cur = best_cfg["azimuths"][s] - base_config["azimuths"][s]
                if off == cur:
                    continue
                cfg = copy.deepcopy(best_cfg)
                cfg["azimuths"][s] = base_config["azimuths"][s] + off
                improved |= consider(cfg, f"azimuth sector{s} -> {off:+}")
        for p in POWERS:
            if p == best_cfg.get("power_offset_db", 0.0):
                continue
            cfg = copy.deepcopy(best_cfg)
            cfg["power_offset_db"] = p
            improved |= consider(cfg, f"power -> {p:+} dB")
        if not improved:
            break
    best_cur = scorer.score(best_cfg, w_current)
    return best_cfg, base_proj, best_proj, best_cur, history
