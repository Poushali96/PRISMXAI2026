"""RC-PRISM: release-pacing and event-aware extensions for the PRISM-XAI v2.0 simulator.

Applied as patches from outside (like the stage-1 duty-cycle patch); the original
simulator file is never edited, so its SHA-256 stays valid.

Diagnosis that motivates this module (from the stage-1 full results):
  * Critical coverage requires zone-mean AoX <= reliability_aox_threshold (8 steps).
    One delivery keeps an endpoint fresh for at most 8 steps, so coverage is bounded
    by roughly 8 * deliveries_per_endpoint / critical_steps_per_zone.
  * PRISM's mean AoX is ~230 steps over a 512-step horizon even though it has 8-22
    releases per endpoint. Evenly spaced releases would give a mean AoX of ~12-34.
    The release budget is spent early and endpoints then stay stale for most of
    the horizon. This is a pacing failure, not a shortage of budget.

Variants (all keep the per-release sigma, the release cap and the epsilon budget
exactly as in the original, so the privacy guarantee is unchanged):
  orig         : original PRISM-XAI behaviour.
  paced        : carry-over token bucket. By step t an endpoint may have used at most
                 C*(t+1)/T + BURST releases. Unused tokens carry forward.
  event_paced  : paced, plus event-aware timing. A critical endpoint releases only
                 when its information is about to expire (AoX >= REFRESH_AGE) or has
                 never been released; non-critical releases must leave RESERVE
                 tokens for future critical periods.
  event_only   : event_paced with no non-critical releases at all.

Gate inputs: step index, horizon length, AoX, release counts and the forecast-derived
criticality flag. These are the same signals the v2.0 public-score variant already
treats as public; no raw attribution is read.

Also provides `periodic_rr_dp`, a corrected round-robin baseline. The original
`periodic_fixed_dp` advances its start index by one endpoint per slot, not by the
number of endpoints served, so on UCI (32 endpoints, about 4 served per slot) each
endpoint is served in bursts of ~4 slots followed by ~28 idle slots.

Every result also gains driver-group fidelity metrics (v2):
  driver_cohen_kappa_critical      : genuine Cohen's kappa between true and reconstructed
                                     driver groups on all critical zone-steps, in [-1, 1].
                                     NaN with driver_kappa_status = 'no_events' or
                                     'degenerate' (chance agreement = 1); never clipped.
  driver_balanced_accuracy_critical: mean per-group recall; robust to the dominant group.
  majority_agreement_ref           : agreement of always guessing the majority group.
  driver_skill_vs_majority_covered : (operational_agreement_covered - ref) / (1 - ref).
                                     This is what v1 wrongly called "driver_kappa_covered".
                                     It is a skill score, NOT kappa: its lower bound is
                                     -ref/(1-ref) (about -30 when ref = 0.97), and it is
                                     NaN when no critical event was covered.

INSTALL ORDER: call install(mod) BEFORE any duty-cycle patch, and add
"periodic_rr_dp" to the duty patch's set of external schedulers, so the throttle
also wraps the corrected round-robin baseline.
"""
from __future__ import annotations

import numpy as np

from typing import Any, Dict

GATED = {"prism_xai", "repaired_mapxai"}
STATE: Dict[str, Any] = {
    "variant": "orig",
    "burst": 1.0,        # tokens available at t=0
    "refresh_age": 7,    # AoX at which a critical endpoint is refreshed (threshold is 8)
    "reserve": 2.0,      # tokens a non-critical release must leave untouched
    "rr_pointer": 0,
}
VARIANTS = ("orig", "paced", "event_paced", "event_only")


def install(mod) -> None:
    """Patch an already-loaded simulator module in place. Idempotent."""
    if getattr(mod, "_RC_PRISM_INSTALLED", False):
        return
    orig_build = mod.build_candidates
    orig_select = mod.select_candidates
    orig_simulate = mod.simulate_scheduler

    def gated_build(scheduler, flags, t, aox, critical_by_endpoint, forecast_novelty,
                    hproxy, delivered_counts, private_release_counts, residual_norm,
                    noise_floor, critical_coverage_so_far, fairness_so_far,
                    accountants, config, trace, d):
        cands = orig_build(scheduler, flags, t, aox, critical_by_endpoint,
                           forecast_novelty, hproxy, delivered_counts,
                           private_release_counts, residual_norm, noise_floor,
                           critical_coverage_so_far, fairness_so_far, accountants,
                           config, trace, d)
        variant = STATE["variant"]
        if variant == "orig" or scheduler not in GATED:
            return cands
        cap = max(1, int(config.max_private_releases_per_endpoint))
        horizon = max(1, int(config.n_steps))
        kept = {}
        for i, options in cands.items():
            used = int(private_release_counts[i])
            tokens = cap * (t + 1) / horizon + STATE["burst"] - used
            critical = bool(critical_by_endpoint[i])
            if variant == "paced":
                ok = tokens >= 1.0
            else:
                if critical:
                    due = used == 0 or float(aox[i]) >= STATE["refresh_age"]
                    ok = tokens >= 1.0 and due
                elif variant == "event_only":
                    ok = False
                else:
                    ok = tokens >= 1.0 + STATE["reserve"]
            if ok:
                kept[i] = options
        return kept

    def rr_select(scheduler, candidates, t, n_endpoints, capacity):
        if scheduler != "periodic_rr_dp":
            return orig_select(scheduler, candidates, t, n_endpoints, capacity)
        selected, used, served = [], 0, 0
        start = STATE["rr_pointer"]
        for offset in range(n_endpoints):
            i = (start + offset) % n_endpoints
            if i not in candidates:
                continue
            option = candidates[i][0]
            if used + option.rbs <= capacity:
                selected.append(option)
                used += option.rbs
                served = offset + 1
        # Next slot starts after the last endpoint served (true round robin).
        STATE["rr_pointer"] = (start + max(served, 1)) % n_endpoints
        return selected

    def rr_build(scheduler, *args, **kwargs):
        # The corrected baseline uses the same candidates as periodic_fixed_dp.
        real = "periodic_fixed_dp" if scheduler == "periodic_rr_dp" else scheduler
        return gated_build(real, *args, **kwargs)

    ref_cache = {}

    def majority_reference(data, outputs, config):
        key = (id(outputs), config.n_zones)
        if key not in ref_cache:
            cal = mod.calibrate_simulation(outputs, data, config)
            crit = mod.criticality_matrix(outputs.pred_eval, data, cal, config)
            groups = [mod.driver_group(outputs.phi_eval[t, data.zones == z].mean(axis=0))
                      for t in range(crit.shape[0]) for z in range(config.n_zones) if crit[t, z]]
            freq = np.bincount(groups, minlength=4) / max(len(groups), 1)
            ref_cache.clear()
            ref_cache[key] = (float(freq.max()), float(freq[0]))
        return ref_cache[key]

    # ---- Driver-group recorder ------------------------------------------------
    # Inside simulate_scheduler, driver_group is called in exactly two places, both
    # as (oracle, reconstruction) pairs: once per zone per step (the agreement used
    # by critical_event_recall), and a second time for non-critical zones only (the
    # mismatch count). Recording the calls gives the per-event confusion data needed
    # for a genuine Cohen's kappa without editing the simulator file.
    orig_driver_group = mod.driver_group
    recorder = {"on": False, "calls": []}

    def recording_driver_group(phi):
        g = orig_driver_group(phi)
        if recorder["on"]:
            recorder["calls"].append(g)
        return g

    mod.driver_group = recording_driver_group

    def parse_calls(calls, crit, n_noncritical_mismatch_reported):
        """Return (oracle, recon) groups on critical zone-steps; validate call order."""
        it, oracle, recon = 0, [], []
        noncrit_mismatch = 0
        for t in range(crit.shape[0]):
            for z in range(crit.shape[1]):
                o, r = calls[it], calls[it + 1]; it += 2
                if crit[t, z]:
                    oracle.append(o); recon.append(r)
                else:
                    o2, r2 = calls[it], calls[it + 1]; it += 2
                    noncrit_mismatch += int(o2 != r2)
        if it != len(calls) or noncrit_mismatch != n_noncritical_mismatch_reported:
            raise RuntimeError(
                "driver_group call pattern differs from the expected simulator code path; "
                "the recorder cannot be trusted for this simulator version.")
        return np.asarray(oracle, int), np.asarray(recon, int)

    def simulate(scheduler, data, outputs, trace, config, *args, **kwargs):
        STATE["rr_pointer"] = 0
        recorder["calls"] = []
        recorder["on"] = True
        try:
            result, frame = orig_simulate(scheduler, data, outputs, trace, config, *args, **kwargs)
        finally:
            recorder["on"] = False
        cal = mod.calibrate_simulation(outputs, data, config)
        crit = mod.criticality_matrix(outputs.pred_eval, data, cal, config)
        oracle, recon = parse_calls(recorder["calls"], crit, result["noncritical_driver_mismatches"])
        recorder["calls"] = []
        result.update(driver_agreement_metrics(oracle, recon))

        ref, zero_ref = majority_reference(data, outputs, config)
        agree = result["operational_agreement_covered"]
        result["majority_agreement_ref"] = ref
        result["zero_recon_agreement_ref"] = zero_ref
        # Skill score relative to always guessing the majority driver group. NOT Cohen's
        # kappa: its range is [-ref/(1-ref), 1], e.g. down to about -30 when ref = 0.97.
        # NaN when no critical event was covered (operational_agreement_covered undefined).
        result["driver_skill_vs_majority_covered"] = (
            (agree - ref) / (1.0 - ref) if np.isfinite(agree) and ref < 1.0 else float("nan"))
        result["rc_variant"] = STATE["variant"] if scheduler in GATED else "none"
        return result, frame

    mod.build_candidates = rr_build
    # Chain onto whatever select_candidates is installed (e.g. the duty-cycle patch).
    mod.select_candidates = rr_select
    mod.simulate_scheduler = simulate
    mod._RC_PRISM_INSTALLED = True


def set_variant(name: str, **params) -> None:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {VARIANTS}")
    STATE["variant"] = name
    for key, value in params.items():
        if key not in STATE:
            raise KeyError(key)
        STATE[key] = value


def cohen_kappa(oracle: np.ndarray, recon: np.ndarray, n_classes: int = 4) -> Dict[str, Any]:
    """Cohen's kappa between true and reconstructed driver groups, always in [-1, 1].

    kappa = (p_o - p_e) / (1 - p_e). Edge cases are explicit, never silently clipped:
      * no events                 -> NaN, reason 'no_events'
      * p_e == 1 (both raters use one identical class for every event): agreement is
        perfect but carries no information beyond chance -> NaN, reason 'degenerate'
    """
    n = len(oracle)
    if n == 0:
        return {"kappa": float("nan"), "p_o": float("nan"), "p_e": float("nan"), "reason": "no_events"}
    p_o = float(np.mean(oracle == recon))
    po_k = np.bincount(oracle, minlength=n_classes) / n
    pr_k = np.bincount(recon, minlength=n_classes) / n
    p_e = float(np.sum(po_k * pr_k))
    if p_e >= 1.0 - 1e-12:
        return {"kappa": float("nan"), "p_o": p_o, "p_e": p_e, "reason": "degenerate"}
    kappa = (p_o - p_e) / (1.0 - p_e)
    assert -1.0 - 1e-9 <= kappa <= 1.0 + 1e-9, kappa
    return {"kappa": float(kappa), "p_o": p_o, "p_e": p_e, "reason": "ok"}


def driver_agreement_metrics(oracle: np.ndarray, recon: np.ndarray) -> Dict[str, Any]:
    """Driver-group fidelity on ALL critical zone-steps (covered or not)."""
    k = cohen_kappa(oracle, recon)
    n = len(oracle)
    per_class_recall = []
    for c in range(4):
        m = oracle == c
        if m.any():
            per_class_recall.append(float(np.mean(recon[m] == c)))
    return {
        "driver_cohen_kappa_critical": k["kappa"],
        "driver_kappa_status": k["reason"],
        "driver_agreement_critical": k["p_o"],
        "driver_chance_agreement_critical": k["p_e"],
        # Balanced accuracy: mean per-class recall over driver groups that occur.
        # Insensitive to the dominant group, unlike raw agreement.
        "driver_balanced_accuracy_critical": float(np.mean(per_class_recall)) if per_class_recall else float("nan"),
        "driver_events_critical": int(n),
    }


def freshness_bound(result: Dict[str, Any], n_zones: int, aox_threshold: float = 8.0) -> float:
    """Approximate upper bound on critical coverage from the delivery count alone."""
    crit_steps_per_zone = max(result["critical_events"] / max(n_zones, 1), 1e-9)
    return min(1.0, aox_threshold * result["mean_updates_per_endpoint"] / crit_steps_per_zone)
