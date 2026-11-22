"""RC-PRISM v3 cell: claims audit of the v2 results + driver-fidelity diagnosis.

Run this AFTER the v2 runner (cell 18) has finished. It never edits rc_results_v2/;
everything new goes to PRISM_XAI_Matched/rc_results_v3_diag/.

Why this cell exists (from rc_results_v2, held-out seeds 2001-2010):
  * driver_cohen_kappa_critical is about 0.00 for EVERY scheduler on both datasets
    (OPSD agreement 0.42-0.50 while always guessing the majority group gives 0.95).
    So critical_event_recall measures coverage times a coin flip, not explanation quality.
  * All 48 supported traffic claims are against baselines whose own coverage is low
    (joint_no_compression, and on UCI channel_fixed_dp / periodic_fixed_dp with ~0-1%
    coverage). PRISM never matched aox_fixed_dp or periodic_rr_dp (75-100% coverage).
  * The v2 REPORT printout was cut before the "supported X/Y" line.

PART A (no simulation): claims audit of the v2 confirmatory results, split by how strong
          the matched baseline is. Writes CLAIMS_AUDIT.md and claims_by_baseline_tier.csv.
PART B (simulation on TUNING seeds only, so held-out/confirm seeds stay clean):
          finds out WHY kappa is ~0. For each scheduler it runs the same schedule twice:
            dp       : exactly as in v2
            no_noise : DP noise set to zero (a non-private upper bound, diagnostic only)
          and records how far the reconstruction drifts from the true explanation
          (recon_norm_ratio). A local synthetic check showed the error-feedback update
          lets DP noise accumulate like a random walk (reconstruction norm 10-200x the
          true explanation norm), because the clipped correction is much smaller than
          the per-release noise. This part checks whether the same happens on OPSD/UCI.
          Writes driver_diagnosis_runs.csv, driver_diagnosis_summary.csv, DRIVER_DIAGNOSIS.md.

Nothing here is a confirmatory result. It decides what the paper may claim and whether a
mechanism change (new tune -> lock -> fresh-seed confirm) is worth doing.
"""
import os, sys, json, hashlib, importlib.util, subprocess
from pathlib import Path

DIAG_SEEDS = [11, 22, 33]          # subset of the TUNING seeds; add more for tighter numbers
DIAG_EPSILONS = [8.0, 16.0, 32.0]
DIAG_CAP, DIAG_THRESHOLD = 8, 5.0   # PRISM setting most often chosen by the v1 lock
RUN_PART_B = True
TEST_MODE = os.environ.get('RC_TEST') == '1'   # local smoke test on synthetic data only

# ---------------- paths (same layout as the v2 runner) ----------------
if os.environ.get('RC_ROOT'):
    DRIVE_ROOT = Path(os.environ['RC_ROOT']); IN_COLAB = False
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_ROOT = Path('/content/drive/MyDrive/PRISM_XAI_Matched'); IN_COLAB = True
    except ImportError:
        DRIVE_ROOT = Path('/mnt/data/PRISM_XAI_Matched'); IN_COLAB = False
STAGE1 = DRIVE_ROOT / 'full_results'
V2 = DRIVE_ROOT / 'rc_results_v2'
OUT = DRIVE_ROOT / ('rc_results_v3_diag_TEST' if TEST_MODE else 'rc_results_v3_diag')
OUT.mkdir(parents=True, exist_ok=True)

if IN_COLAB:
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'lightgbm', 'shap'])
import numpy as np
import pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_atomic(path, text):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(text); os.replace(tmp, path)

# =====================================================================
# PART A: claims audit of v2 (reads files only)
# =====================================================================
def tier(cov):
    if not np.isfinite(cov): return 'unknown'
    if cov < 0.05: return 'degenerate (<5% coverage)'
    if cov < 0.50: return 'weak (5-50% coverage)'
    return 'strong (>=50% coverage)'

def part_a():
    conf = pd.read_csv(V2 / 'confirmatory_vs_baselines.csv')
    held = pd.read_csv(V2 / 'heldout' / 'all_run_metrics.csv', low_memory=False)
    lock = json.loads((V2 / 'selection_lock.json').read_text())
    keys = ['dataset', 'xai_method', 'epsilon_budget', 'scheduler', 'setting']
    base_cov = (held[held.setting_kind == 'BASELINE'].groupby(keys).critical_coverage.mean()
                .rename('baseline_coverage').reset_index().rename(columns={'setting': 'baseline_setting'}))
    best_cov = (held[held.setting_kind == 'BASELINE'].groupby(['dataset', 'xai_method', 'epsilon_budget', 'scheduler'])
                .critical_coverage.max().rename('scheduler_best_coverage').reset_index())
    c = conf.merge(base_cov, on=['dataset', 'xai_method', 'epsilon_budget', 'scheduler', 'baseline_setting'], how='left')
    c = c.merge(best_cov, on=['dataset', 'xai_method', 'epsilon_budget', 'scheduler'], how='left')
    c['baseline_tier'] = [tier(x) for x in c.baseline_coverage]
    c['scheduler_tier'] = [tier(x) for x in c.scheduler_best_coverage]
    c.to_csv(OUT / 'claims_with_baseline_coverage.csv', index=False)

    t = c[c.status == 'tested']
    by_tier = (t.groupby(['dataset', 'baseline_tier'])
               .agg(tested=('claim_supported', 'size'), supported=('claim_supported', 'sum'),
                    mean_traffic_reduction_pct=('relative_reduction_pct', 'mean'),
                    mean_baseline_coverage=('baseline_coverage', 'mean')).reset_index())
    by_tier.to_csv(OUT / 'claims_by_baseline_tier.csv', index=False)
    untested = (c[c.status != 'tested'].groupby(['dataset', 'scheduler_tier', 'scheduler']).size()
                .rename('untested_comparisons').reset_index())

    # Driver fidelity on held-out runs: Cohen kappa vs chance, balanced accuracy vs 1/groups
    held['group'] = np.where(held.setting_kind == 'PRISM', 'PRISM_' + held.variant.astype(str), held.scheduler)
    fid = (held.groupby(['dataset', 'group'])
           [['driver_cohen_kappa_critical', 'driver_balanced_accuracy_critical', 'driver_agreement_critical',
             'majority_agreement_ref', 'critical_coverage']].mean().round(3).reset_index())
    fid.to_csv(OUT / 'driver_fidelity_heldout_v2.csv', index=False)

    n_t = len(t); n_ok = int(t.claim_supported.sum())
    strong = t[t.baseline_tier.str.startswith('strong')]
    strong_desc = ', '.join(sorted({f'{r.dataset} {r.scheduler} eps={r.epsilon_budget:g}' for r in strong.itertuples()})) or 'none'
    degen = c[c.scheduler_tier.str.startswith('degenerate')][['dataset', 'scheduler']].drop_duplicates()
    degen_desc = ', '.join(f'{r.dataset} {r.scheduler}' for r in degen.itertuples()) or 'none'
    never = c.groupby(['dataset', 'scheduler']).status.apply(lambda s: (s == 'tested').sum() == 0)
    never_strong = [f'{d} {s}' for (d, s), v in never.items() if v and
                    c[(c.dataset == d) & (c.scheduler == s)].scheduler_tier.iloc[0].startswith('strong')]
    lines = ['# Claims audit of rc_results_v2 (held-out seeds 2001-2010)', '',
             f'Lock SHA-256: {sha(V2 / "selection_lock.json")}',
             f'Traffic claims vs baselines: supported in {n_ok}/{n_t} locked comparisons '
             f'({len(c) - n_t} further comparisons untested because no tuning match existed).', '',
             '## Supported claims by how strong the matched baseline is', '',
             by_tier.round(3).to_markdown(index=False), '',
             f'Supported claims against a baseline with >=50% coverage: {int(strong.claim_supported.sum())}/{len(strong)}.', '',
             '## Comparisons that could not be tested (no PRISM setting matched the baseline on tuning seeds)', '',
             untested.to_markdown(index=False), '',
             '## Driver-group fidelity on held-out runs', '',
             'Cohen kappa near 0 means the reconstructed driver group agrees with the truth no more than chance.',
             'Balanced accuracy at chance is 1/(number of driver groups that occur).', '',
             fid.to_markdown(index=False), '',
             '## What this means for the paper', '',
             f'- Supported claims against a strong (>=50% coverage) baseline setting: {int(strong.claim_supported.sum())}/{len(strong)}'
             f' ({strong_desc}). Every other supported claim is against a weak or degenerate baseline setting.',
             f'- Strong schedulers PRISM was never matched against (no tuning match at all): {", ".join(never_strong) or "none"}.',
             '  Do not describe PRISM as saving traffic versus these.',
             f'- Schedulers whose best coverage is <5% (report as non-functional here, not as wins): {degen_desc}.',
             '- critical_event_recall and any driver/explanation-fidelity claim are not supported while',
             '  kappa is ~0 for every method. Part B tests why.',
             '- The pacing result (event_paced vs orig coverage) stands on its own; see confirmatory_ablation_vs_orig.csv.']
    write_atomic(OUT / 'CLAIMS_AUDIT.md', '\n'.join(lines))
    print('\n'.join(lines[:12]))
    print(by_tier.round(3).to_string(index=False))

if not TEST_MODE:
    print('== PART A: claims audit =='); part_a()

# =====================================================================
# PART B: why is driver kappa ~0? (tuning seeds only)
# =====================================================================
if RUN_PART_B:
    sim_path = Path(os.environ['SIM_PATH']) if os.environ.get('SIM_PATH') else None
    if sim_path is None:
        found = list((STAGE1 / 'repository').rglob('code/prism_xai_icc2027_v2_0_public_score_experimental.py'))
        if len(found) != 1:
            raise RuntimeError('Simulator not found under full_results/repository (run the stage-1 runner first).')
        sim_path = found[0]
    rc_path = DRIVE_ROOT / 'rc_prism.py'
    SIM_SHA, RC_SHA = sha(sim_path), sha(rc_path)
    if not TEST_MODE:
        v2sig = json.loads((V2 / 'signature.json').read_text())
        if v2sig['sim_sha256'] != SIM_SHA or v2sig['rc_sha256'] != RC_SHA:
            raise RuntimeError('Simulator or rc_prism.py differ from the files used for rc_results_v2.')
    cache_dir = Path(os.environ.get('CACHE_DIR', STAGE1 / 'data_cache'))

    # Fresh module objects, so patches from earlier cells in this runtime cannot leak in.
    spec = importlib.util.spec_from_file_location('prism_v3diag', sim_path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    spec2 = importlib.util.spec_from_file_location('rc_prism_v3diag', rc_path)
    rc = importlib.util.module_from_spec(spec2); sys.modules[spec2.name] = rc; spec2.loader.exec_module(rc)
    rc.install(mod)

    # --- diagnostic patches (both are no-ops at their defaults) ---
    KNOB = {'noise_scale': 1.0, 'recon': None}
    _privatize = mod.privatize_and_compress
    def privatize_diag(delta, clip_norm, sensitivity, mode, base_noise):
        return _privatize(delta, clip_norm, sensitivity, mode, base_noise * KNOB['noise_scale'])
    mod.privatize_and_compress = privatize_diag
    _attack = mod.attribute_attack_auc          # called once per run with the full recon history
    def attack_capture(recon_history, labels, seed):
        KNOB['recon'] = np.array(recon_history, copy=True)
        return _attack(recon_history, labels, seed)
    mod.attribute_attack_auc = attack_capture

    DATASETS = ['synthetic'] if TEST_MODE else ['opsd', 'uci']
    XAI = ['feature_ablation'] if TEST_MODE else ['treeshap', 'feature_ablation']
    SEEDS = [11, 22] if TEST_MODE else DIAG_SEEDS
    EPS = [16.0] if TEST_MODE else DIAG_EPSILONS
    RUNS = [('prism_xai', v) for v in ('orig', 'paced', 'event_paced')] + \
           [(b, 'baseline') for b in ('periodic_fixed_dp', 'periodic_rr_dp', 'channel_fixed_dp',
                                      'aox_fixed_dp', 'joint_no_compression')]

    def base_config(dataset):
        cfg = mod.ExperimentConfig.full_defaults()
        cfg.dataset = dataset; cfg.cache_dir = str(cache_dir); cfg.models = ['tree']; cfg.run_ablations = False
        if TEST_MODE:
            cfg.preset = 'quick'; cfg.n_steps = 192; cfg.calibration_steps = 96
            cfg.n_endpoints = 12; cfg.n_zones = 3; cfg.train_max_rows = 20000
        else:   # identical to the v2 runner
            cfg.n_steps = 512; cfg.calibration_steps = 96; cfg.train_max_rows = 80000; cfg.tree_estimators = 500
            cfg.n_endpoints = 11 if dataset == 'opsd' else 32
            cfg.n_zones = 3 if dataset == 'opsd' else 4
        cfg.max_private_releases_per_endpoint = DIAG_CAP
        cfg.public_release_score_threshold = DIAG_THRESHOLD
        return cfg

    def driver_from_recon(recon, data, outputs, cfg):
        """Recompute (oracle, recon) driver groups on critical zone-steps from the captured history."""
        cal = mod.calibrate_simulation(outputs, data, cfg)
        crit = mod.criticality_matrix(outputs.pred_eval, data, cal, cfg)
        oracle, rec, r_norm, o_norm = [], [], [], []
        for t in range(crit.shape[0]):
            for z in range(cfg.n_zones):
                idx = data.zones == z
                oz, rz = outputs.phi_eval[t, idx].mean(axis=0), recon[t, idx].mean(axis=0)
                if crit[t, z]:
                    oracle.append(mod.driver_group(oz)); rec.append(mod.driver_group(rz))
                if t >= int(0.9 * crit.shape[0]):
                    r_norm.append(np.linalg.norm(rz)); o_norm.append(np.linalg.norm(oz))
        return (np.asarray(oracle, int), np.asarray(rec, int),
                float(np.mean(r_norm) / max(np.mean(o_norm), 1e-12)))

    def run_cell(dataset, seed, xai):
        path = OUT / 'cells' / f'{dataset}_seed{seed}_{xai}.csv'
        mark = path.with_suffix('.done.json')
        if path.is_file() and mark.is_file() and json.loads(mark.read_text()).get('sha256') == sha(path):
            print('  SKIP', path.name, flush=True); return pd.read_csv(path)
        print('  RUN ', path.name, flush=True)
        cfg = base_config(dataset)
        mod.set_global_seed(seed)
        data = mod.prepare_data(cfg, seed)
        trace = mod.make_network_trace(cfg, data, seed)
        outputs = mod.train_and_explain(data, cfg, 'tree', seed, xai)
        rows = []
        for eps in EPS:
            for scheduler, variant in RUNS:
                rc.set_variant(variant if scheduler == 'prism_xai' else 'orig')
                for noise in ('dp', 'no_noise'):
                    KNOB['noise_scale'] = 1.0 if noise == 'dp' else 0.0
                    res, _ = mod.simulate_scheduler(scheduler, data, outputs, trace, cfg, eps, seed)
                    oracle, rec, ratio = driver_from_recon(KNOB['recon'], data, outputs, cfg)
                    k = rc.cohen_kappa(oracle, rec)
                    # The captured history must reproduce the simulator's own kappa exactly.
                    if not np.isclose(k['kappa'], res['driver_cohen_kappa_critical'], atol=1e-9, equal_nan=True):
                        raise RuntimeError('captured reconstruction does not reproduce the simulator kappa')
                    rows.append({'dataset': dataset, 'xai_method': outputs.explainer_backend, 'seed': seed,
                                 'epsilon_budget': eps, 'scheduler': scheduler, 'variant': variant, 'noise': noise,
                                 'cohen_kappa': k['kappa'], 'agreement': k['p_o'], 'chance_agreement': k['p_e'],
                                 'balanced_accuracy': res['driver_balanced_accuracy_critical'],
                                 'majority_agreement_ref': res['majority_agreement_ref'],
                                 'n_groups_present': int(len(np.unique(oracle))),
                                 'critical_coverage': res['critical_coverage'],
                                 'mean_updates_per_endpoint': res['mean_updates_per_endpoint'],
                                 'epsilon_spent_max': res['epsilon_spent_max'],
                                 'recon_norm_ratio_end': ratio})
        KNOB['noise_scale'] = 1.0
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp'); df.to_csv(tmp, index=False); os.replace(tmp, path)
        mark.write_text(json.dumps({'sha256': sha(path), 'sim_sha256': SIM_SHA, 'rc_sha256': RC_SHA}))
        return df

    print('== PART B: driver-fidelity diagnosis (tuning seeds', SEEDS, ') ==')
    runs = pd.concat([run_cell(ds, s, x) for ds in DATASETS for s in SEEDS for x in XAI], ignore_index=True)
    runs.to_csv(OUT / 'driver_diagnosis_runs.csv', index=False)

    runs['method'] = np.where(runs.scheduler == 'prism_xai', 'PRISM_' + runs.variant, runs.scheduler)
    summ = (runs.pivot_table(index=['dataset', 'method'], columns='noise',
                             values=['cohen_kappa', 'balanced_accuracy', 'recon_norm_ratio_end'], aggfunc='mean'))
    summ.columns = [f'{m}_{n}' for m, n in summ.columns]
    summ = summ.reset_index()
    summ['chance_balanced_accuracy'] = summ.dataset.map(
        1.0 / runs.groupby('dataset').n_groups_present.median())
    summ.to_csv(OUT / 'driver_diagnosis_summary.csv', index=False)

    # Interpretation rules, fixed here before looking at the output.
    def verdict(r):
        if r.cohen_kappa_no_noise < 0.10:
            return 'no driver signal even without noise: schedule/staleness (or the metric) is the limit'
        if r.cohen_kappa_dp < 0.05:
            if r.recon_norm_ratio_end_dp > 3.0:
                return 'DP noise accumulates (random walk in the reconstruction): the update rule is the limit'
            return 'DP noise swamps the signal at these epsilons: the per-release SNR is the limit'
        return 'driver signal survives DP'
    summ['verdict'] = summ.apply(verdict, axis=1)
    lines = ['# Driver-fidelity diagnosis (TUNING seeds only; diagnostic, not confirmatory)', '',
             f'Seeds {SEEDS}; epsilons {EPS}; PRISM cap {DIAG_CAP}, threshold {DIAG_THRESHOLD:g}; baselines at duty 1.',
             'no_noise = same scheduler with the DP noise set to zero. It is NOT private; it only shows',
             'how much driver information the schedule and update rule could carry without noise.',
             'recon_norm_ratio_end = mean |reconstructed zone explanation| / |true zone explanation| over the',
             'last 10% of steps. Values far above 1 mean noise has accumulated in the reconstruction.', '',
             summ.round(3).to_markdown(index=False), '',
             'Per-epsilon detail is in driver_diagnosis_runs.csv.']
    write_atomic(OUT / 'DRIVER_DIAGNOSIS.md', '\n'.join(lines))
    print('\n'.join(lines))
print('RESULTS:', OUT)
