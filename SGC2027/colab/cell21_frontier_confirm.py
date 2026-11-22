"""RC-PRISM cell 21: PRE-REGISTERED confirmation of the frontier result on fresh seeds 3001-3010.

Why: in the paper (Fig. "Traffic-coverage operating points") event-paced PRISM lies above the
best throttled-baseline coverage achievable at no more traffic in 98% (OPSD) / 100% (UCI) of
held-out runs. That analysis was NOT pre-registered (seeds 2001-2010 had been inspected), so the
paper labels it "descriptive". This cell turns it into a confirmatory result on unused seeds.

The hypothesis, test and decision rule are written to frontier_prereg.json BEFORE any run and
the cell refuses to run if that file is missing or changed. Seeds 3001-3010 were reserved by the
v3 mechanism lock and never used (that lock was NO-GO), so they are fresh.

Hypothesis H_F, one per configuration c = (dataset, XAI, epsilon, cap M, threshold), 108 in total:
  margin_s(c) = coverage of event-paced PRISM(c) on seed s
              - max coverage over all throttled baselines (5 schedulers x 7 duty cycles) on the
                same seed, dataset, XAI and epsilon whose traffic <= PRISM(c)'s traffic on seed s
                (if none, the lowest-traffic baseline point)
  H_F(c): mean_s margin_s(c) > 0. One-sided one-sample t-test over the 10 seeds (Wilcoxon
  reported), Holm correction over all 108 configurations, alpha = 0.05.
Reported: number of configurations with Holm-significant positive margin, median margin, and the
fraction of all seed-level comparisons with margin > 0.

Cost: one model fit per (dataset, seed, XAI) = 40 fits; 132 simulations per fit
(3 epsilons x (9 PRISM settings + 35 baseline points)). Resumable per cell.
Outputs: PRISM_XAI_Matched/rc_results_v4_frontier/  (FRONTIER_REPORT.md, frontier_confirm_macros.tex)
Send me FRONTIER_REPORT.md and frontier_confirm_macros.tex; the paper then upgrades the claim.
"""
import os, sys, json, hashlib, importlib.util, subprocess, datetime as dt
from pathlib import Path

SEEDS = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010]
DATASETS = ['opsd', 'uci']
XAI_METHODS = ['treeshap', 'feature_ablation']
EPSILONS = [8.0, 16.0, 32.0]
CAPS = [8, 16, 32]
THRESHOLDS = [1.0, 3.0, 5.0]
BASELINES = ['periodic_fixed_dp', 'periodic_rr_dp', 'channel_fixed_dp', 'aox_fixed_dp', 'joint_no_compression']
DUTIES = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
ALPHA = 0.05
TEST_MODE = os.environ.get('RC_TEST') == '1'

PREREG = {
    'hypothesis': 'H_F(c): mean over seeds of [coverage(event-paced PRISM, c) - best throttled-baseline '
                  'coverage at <= PRISM traffic, same seed/dataset/XAI/epsilon] > 0',
    'envelope': 'max coverage over 5 baselines x 7 duty cycles with traffic <= PRISM traffic; '
                'if none, coverage of the lowest-traffic baseline point',
    'test': 'one-sided one-sample t over 10 seeds per configuration; Wilcoxon reported; Holm over 108',
    'alpha': ALPHA, 'seeds': SEEDS, 'datasets': DATASETS, 'xai': XAI_METHODS, 'epsilons': EPSILONS,
    'caps': CAPS, 'thresholds': THRESHOLDS, 'baselines': BASELINES, 'duties': DUTIES,
    'prism_variant': 'event_paced', 'protocol_version': 'v4-frontier-1',
}

if os.environ.get('RC_ROOT'):
    DRIVE_ROOT = Path(os.environ['RC_ROOT']); IN_COLAB = False
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_ROOT = Path('/content/drive/MyDrive/PRISM_XAI_Matched'); IN_COLAB = True
    except ImportError:
        DRIVE_ROOT = Path('/mnt/data/PRISM_XAI_Matched'); IN_COLAB = False
STAGE1, V2, V3M = DRIVE_ROOT / 'full_results', DRIVE_ROOT / 'rc_results_v2', DRIVE_ROOT / 'rc_results_v3_mech'
OUT = DRIVE_ROOT / ('rc_results_v4_frontier_TEST' if TEST_MODE else 'rc_results_v4_frontier')
OUT.mkdir(parents=True, exist_ok=True)
if IN_COLAB:
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'lightgbm', 'shap'])
import warnings; warnings.filterwarnings('ignore', message='X does not have valid feature names')
import numpy as np
import pandas as pd
from scipy import stats

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if TEST_MODE:
    SEEDS, DATASETS, XAI_METHODS, EPSILONS = [3001, 3002, 3003], ['synthetic'], ['feature_ablation'], [16.0]
    CAPS, THRESHOLDS, DUTIES = [8, 32], [3.0], [1.0, 0.25, 0.0625]
    PREREG.update(seeds=SEEDS, datasets=DATASETS, xai=XAI_METHODS, epsilons=EPSILONS, caps=CAPS,
                  thresholds=THRESHOLDS, duties=DUTIES, test_mode=True)

# ---------- 0. seeds must be fresh; pre-registration written before any run ----------
if not TEST_MODE:
    used = set(json.loads((V2 / 'signature.json').read_text())['heldout_seeds'])
    if set(SEEDS) & used:
        raise RuntimeError('Seeds overlap the v2 held-out seeds.')
    if (V3M / 'confirm').exists():
        raise RuntimeError('rc_results_v3_mech/confirm exists: seeds 3001-3010 may have been used.')
pre_path = OUT / 'frontier_prereg.json'
if pre_path.exists():
    if json.loads(pre_path.read_text())['prereg'] != PREREG:
        raise RuntimeError('Pre-registration differs from this cell. Do not edit the hypothesis after writing it.')
else:
    if (OUT / 'cells').exists() and any((OUT / 'cells').iterdir()):
        raise RuntimeError('Runs exist without a pre-registration file.')
    pre_path.write_text(json.dumps({'prereg': PREREG, 'created': dt.datetime.now().isoformat(timespec='seconds')}, indent=2))
PRE_SHA = sha(pre_path)
print('Pre-registration SHA-256 (record it):', PRE_SHA)

# ---------- 1. simulator + patches, identical to the v2 runner ----------
sim_path = Path(os.environ['SIM_PATH']) if os.environ.get('SIM_PATH') else None
if sim_path is None:
    found = list((STAGE1 / 'repository').rglob('code/prism_xai_icc2027_v2_0_public_score_experimental.py'))
    if len(found) != 1:
        raise RuntimeError('Simulator not found under full_results/repository.')
    sim_path = found[0]
rc_path = DRIVE_ROOT / 'rc_prism.py'
SIM_SHA, RC_SHA = sha(sim_path), sha(rc_path)
if not TEST_MODE:
    v2sig = json.loads((V2 / 'signature.json').read_text())
    if v2sig['sim_sha256'] != SIM_SHA or v2sig['rc_sha256'] != RC_SHA:
        raise RuntimeError('Simulator or rc_prism.py differ from those used for rc_results_v2.')
cache_dir = Path(os.environ.get('CACHE_DIR', STAGE1 / 'data_cache'))
spec = importlib.util.spec_from_file_location('prism_v4front', sim_path)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
spec2 = importlib.util.spec_from_file_location('rc_prism_v4front', rc_path)
rc = importlib.util.module_from_spec(spec2); sys.modules[spec2.name] = rc; spec2.loader.exec_module(rc)
rc.install(mod)                                  # before the duty patch, as in v2
EXTERNAL = set(BASELINES); DUTY = {'v': 1.0}
_select = mod.select_candidates
def duty_selection(scheduler, candidates, t, n_endpoints, capacity):
    if scheduler in EXTERNAL and DUTY['v'] < 1.0:
        if (int(t) * 7919 % 10000) >= round(DUTY['v'] * 10000):
            return []
    return _select(scheduler, candidates, t, n_endpoints, capacity)
mod.select_candidates = duty_selection

def base_config(dataset):
    cfg = mod.ExperimentConfig.full_defaults()
    cfg.dataset = dataset; cfg.cache_dir = str(cache_dir); cfg.models = ['tree']; cfg.run_ablations = False
    if TEST_MODE:
        cfg.preset = 'quick'; cfg.n_steps = 192; cfg.calibration_steps = 96; cfg.n_endpoints = 12
        cfg.n_zones = 3; cfg.train_max_rows = 20000
    else:
        cfg.n_steps = 512; cfg.calibration_steps = 96; cfg.train_max_rows = 80000; cfg.tree_estimators = 500
        cfg.n_endpoints = 11 if dataset == 'opsd' else 32
        cfg.n_zones = 3 if dataset == 'opsd' else 4
    return cfg

SIG = {'prereg_sha256': PRE_SHA, 'sim_sha256': SIM_SHA, 'rc_sha256': RC_SHA}

def run_cell(dataset, seed, xai):
    path = OUT / 'cells' / f'{dataset}_seed{seed}_{xai}.csv'
    mark = path.with_suffix('.done.json')
    if path.is_file() and mark.is_file():
        m = json.loads(mark.read_text())
        if m.get('sha256') == sha(path) and m.get('sig') == SIG:
            print('  SKIP', path.name, flush=True); return pd.read_csv(path)
    print('  RUN ', path.name, flush=True)
    cfg = base_config(dataset)
    mod.set_global_seed(seed)
    data = mod.prepare_data(cfg, seed)
    trace = mod.make_network_trace(cfg, data, seed)
    out = mod.train_and_explain(data, cfg, 'tree', seed, xai)
    rows = []
    for eps in EPSILONS:
        DUTY['v'] = 1.0; rc.set_variant('event_paced')
        for cap in CAPS:
            for thr in THRESHOLDS:
                cfg.max_private_releases_per_endpoint = cap; cfg.public_release_score_threshold = thr
                res, _ = mod.simulate_scheduler('prism_xai', data, out, trace, cfg, eps, seed)
                rows.append(dict(kind='PRISM', scheduler='prism_xai', cap=cap, threshold=thr, duty=np.nan,
                                 kbit=res['transmitted_kbits'], coverage=res['critical_coverage'],
                                 eps_spent=res['epsilon_spent_max'], dataset=dataset, seed=seed,
                                 xai_method=out.explainer_backend, epsilon_budget=eps))
        rc.set_variant('orig'); cfg.max_private_releases_per_endpoint = 8; cfg.public_release_score_threshold = 3.0
        for duty in DUTIES:
            DUTY['v'] = duty
            for b in BASELINES:
                res, _ = mod.simulate_scheduler(b, data, out, trace, cfg, eps, seed)
                rows.append(dict(kind='BASELINE', scheduler=b, cap=np.nan, threshold=np.nan, duty=duty,
                                 kbit=res['transmitted_kbits'], coverage=res['critical_coverage'],
                                 eps_spent=res['epsilon_spent_max'], dataset=dataset, seed=seed,
                                 xai_method=out.explainer_backend, epsilon_budget=eps))
        DUTY['v'] = 1.0
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); df.to_csv(tmp, index=False); os.replace(tmp, path)
    mark.write_text(json.dumps({'sha256': sha(path), 'sig': SIG}))
    return df

runs = pd.concat([run_cell(d, s, x) for d in DATASETS for s in SEEDS for x in XAI_METHODS], ignore_index=True)
runs.to_csv(OUT / 'all_runs.csv', index=False)

# ---------- 2. confirmatory analysis exactly as pre-registered ----------
def envelope(B, T):
    ok = B[B.kbit <= T]
    if len(ok): return ok.coverage.max()
    return B[B.kbit == B.kbit.min()].coverage.max()
mrows = []
for (d, x, e, s), g in runs.groupby(['dataset', 'xai_method', 'epsilon_budget', 'seed']):
    B = g[g.kind == 'BASELINE']
    for _, p in g[g.kind == 'PRISM'].iterrows():
        mrows.append(dict(dataset=d, xai_method=x, epsilon_budget=e, seed=s, cap=p.cap, threshold=p.threshold,
                          kbit=p.kbit, coverage=p.coverage, envelope=envelope(B, p.kbit),
                          margin=p.coverage - envelope(B, p.kbit)))
M = pd.DataFrame(mrows); M.to_csv(OUT / 'frontier_margins.csv', index=False)

def holm(p):
    p = np.asarray(p, float); adj = np.full_like(p, np.nan); ok = np.isfinite(p)
    idx = np.where(ok)[0]; order = idx[np.argsort(p[idx])]; run = 0.0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (len(idx) - r) * p[j])); adj[j] = run
    return adj
tests = []
for key, g in M.groupby(['dataset', 'xai_method', 'epsilon_budget', 'cap', 'threshold']):
    m = g.margin.to_numpy()
    if len(m) != len(SEEDS): raise RuntimeError(f'incomplete seeds for {key}')
    if np.allclose(m, m[0]): p = 0.0 if m[0] > 0 else 1.0
    else: p = stats.ttest_1samp(m, 0, alternative='greater').pvalue
    try: pw = stats.wilcoxon(m, alternative='greater').pvalue if np.any(m != 0) else 1.0
    except ValueError: pw = np.nan
    tests.append(dict(zip(['dataset', 'xai_method', 'epsilon_budget', 'cap', 'threshold'], key),
                      n_seeds=len(m), mean_margin=m.mean(), median_margin=np.median(m), p_t=p, p_wilcoxon=pw))
T = pd.DataFrame(tests); T['p_holm'] = holm(T.p_t); T['supported'] = (T.p_holm < ALPHA) & (T.mean_margin > 0)
T.to_csv(OUT / 'frontier_tests.csv', index=False)

mac = {}
for d in DATASETS:
    t = T[T.dataset == d]; mm = M[M.dataset == d]; tag = d.upper()
    mac[f'fcSup{tag}'] = int(t.supported.sum()); mac[f'fcN{tag}'] = len(t)
    mac[f'fcAbove{tag}'] = f'{100 * (mm.margin > 0).mean():.1f}'
    mac[f'fcMed{tag}'] = f'{100 * mm.margin.median():+.1f}'
mac['fcPreSha'] = PRE_SHA[:12]
(OUT / 'frontier_confirm_macros.tex').write_text(
    '% Generated by cell 21 (pre-registered frontier confirmation, fresh seeds 3001-3010)\n' +
    '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items()) + '\n')
rep = ['# Pre-registered frontier confirmation (fresh seeds)', '',
       f'Pre-registration SHA-256: {PRE_SHA}', f'Seeds: {SEEDS}', '', PREREG['hypothesis'], '']
for d in DATASETS:
    tag = d.upper()
    rep.append(f'{tag}: H_F supported (Holm) in {mac[f"fcSup{tag}"]}/{mac[f"fcN{tag}"]} configurations; '
               f'margin > 0 in {mac[f"fcAbove{tag}"]}% of seed-level comparisons; median margin '
               f'{mac[f"fcMed{tag}"]} coverage points.')
rep += ['', T.round(4).to_markdown(index=False)]
(OUT / 'FRONTIER_REPORT.md').write_text('\n'.join(rep))
print('\n'.join(rep[:10]))
print('RESULTS:', OUT)
