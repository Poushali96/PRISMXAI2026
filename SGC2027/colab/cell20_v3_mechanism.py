"""RC-PRISM v3 cell: redesigned explanation release, tuned on tuning seeds, go/no-go lock,
then (only if GO and you allow it) confirmation on fresh seeds 3001-3010.

Run AFTER cell 19. Never edits rc_results_v2/ or rc_results_v3_diag/.
New outputs go to PRISM_XAI_Matched/rc_results_v3_mech/.

Problem being fixed (cell 19, tuning seeds, OPSD/UCI):
  * DP baselines carry driver information without noise (OPSD kappa 0.41-0.53), but under DP
    their reconstruction drifts to 100-200x the true explanation and kappa falls to ~0.
    Cause: the error-feedback update sends a CLIPPED change; the DP noise added each time is
    much larger than the clip radius, so later releases cannot cancel it (random walk).
  * PRISM has kappa ~0 even without noise: few releases plus a residual buffer that decays
    by 0.65 per step, so the attribution LEVEL is never delivered.

Redesigned release (same Gaussian mechanism, same sigma, same accountant charge per release,
same packet sizes, same scheduler decisions; applied identically to PRISM and all baselines):
  payload = 'level12'  : send the clipped attribution LEVEL phi_t (12 values) instead of the
                         clipped change; Top-k and quantization as in MAP-XAI.
  payload = 'group4'   : send the 4 driver-group magnitudes u_g = sum_{j in g} |phi_j|
                         (the quantity the operator's driver decision uses), clipped.
  server update        : running average of received noisy payloads,
                         est <- est + g * (y - est),  g = max(1/(k+1), g_min),
                         k = releases already received from that endpoint. g_min = 1 means
                         "keep only the latest release". This is post-processing, so the
                         privacy guarantee is unchanged, and it cannot drift: noise averages out.
Privacy note: the clipped level has L2 norm <= C_level, so sensitivity is C_level and the
noise std is sigma * C_level, exactly as the accountant assumes. A level release also needs no
reference attribution, which removes the "reference must be public" condition in the paper's
MAP-XAI section. C_level comes from the calibration window, like the existing clip norms.
Limitation: with group4 the operator receives group magnitudes only, so per-feature Top-k
overlap and cosine are not meaningful for that payload (report driver metrics instead).
Packet sizes are left at the 12-feature MAP-XAI sizes, which overstates group4 traffic.

PHASES (set PHASE):
  'tune'    : every mechanism config x scheduler x epsilon on TUNING seeds.
  'lock'    : choose ONE config with the rule below and decide GO / NO-GO. Writes
              mechanism_lock.json once; never overwrites it.
  'confirm' : fresh seeds 3001-3010, control vs locked config. Refuses to run unless the lock
              says GO and CONFIRM_ALLOWED = True (set it by hand after reading the lock).
Selection rule (fixed before any run):
  For each config (payload, g_min) and PRISM cap, use PRISM event_paced runs only:
    eligible if, on EACH dataset, mean critical_coverage >= control coverage (same cap) - 0.02;
    score = mean over datasets of mean Cohen kappa (dp).
  Pick the highest score. GO if its mean kappa >= 0.10 on OPSD AND >= 0.05 on UCI.
Confirmatory hypotheses (fresh seeds; seeds are the units; Holm over the 12 PRISM cells
dataset x XAI x epsilon):
  H1 kappa(locked) - kappa(control) > 0          (one-sided paired t; Wilcoxon reported)
  H2 kappa(locked) > 0                            (one-sided t)
  H3 coverage(locked) - coverage(control) > -0.02 (lower one-sided 95% bound; non-inferiority)
"""
import os, sys, json, hashlib, importlib.util, subprocess, datetime as dt
from pathlib import Path

PHASE = 'tune'                 # 'tune' -> 'lock' -> (read the lock) -> 'confirm'
CONFIRM_ALLOWED = False        # set True by hand only after reading mechanism_lock.json
TUNE_SEEDS = [11, 22, 33]      # tuning seeds only (same ones cell 19 used)
CONFIRM_SEEDS = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010]
EPSILONS = [8.0, 16.0, 32.0]
PRISM_CAPS = [8, 32]
PRISM_THRESHOLD = 5.0
PAYLOADS = ['level12', 'group4']
G_MINS = [1.0, 0.5, 0.25, 0.1]
LEVEL_CLIP_QUANTILE = 0.95
COV_TOL, GO_KAPPA_OPSD, GO_KAPPA_UCI, NI_MARGIN, ALPHA = 0.02, 0.10, 0.05, 0.02, 0.05
TEST_MODE = os.environ.get('RC_TEST') == '1'
PHASE = os.environ.get('RC_PHASE', PHASE)
if TEST_MODE:   # local smoke test only: synthetic data, tiny grid
    TUNE_SEEDS, CONFIRM_SEEDS, EPSILONS = [11, 22], [3001, 3002, 3003], [16.0]
    CONFIRM_ALLOWED = os.environ.get('RC_CONFIRM') == '1'

# ---------------- paths ----------------
if os.environ.get('RC_ROOT'):
    DRIVE_ROOT = Path(os.environ['RC_ROOT']); IN_COLAB = False
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_ROOT = Path('/content/drive/MyDrive/PRISM_XAI_Matched'); IN_COLAB = True
    except ImportError:
        DRIVE_ROOT = Path('/mnt/data/PRISM_XAI_Matched'); IN_COLAB = False
STAGE1, V2 = DRIVE_ROOT / 'full_results', DRIVE_ROOT / 'rc_results_v2'
OUT = DRIVE_ROOT / ('rc_results_v3_mech_TEST' if TEST_MODE else 'rc_results_v3_mech')
OUT.mkdir(parents=True, exist_ok=True)
LOCK = OUT / 'mechanism_lock.json'

if IN_COLAB:
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'lightgbm', 'shap'])
import warnings; warnings.filterwarnings('ignore', message='X does not have valid feature names')
import numpy as np
import pandas as pd
from scipy import stats

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_atomic(path, text):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(text); os.replace(tmp, path)

# ---------------- load simulator + rc_prism (fresh module objects) ----------------
sim_path = Path(os.environ['SIM_PATH']) if os.environ.get('SIM_PATH') else None
if sim_path is None:
    found = list((STAGE1 / 'repository').rglob('code/prism_xai_icc2027_v2_0_public_score_experimental.py'))
    if len(found) != 1:
        raise RuntimeError('Simulator not found under full_results/repository.')
    sim_path = found[0]
rc_path = DRIVE_ROOT / 'rc_prism.py'
SIM_SHA, RC_SHA, CELL_VERSION = sha(sim_path), sha(rc_path), 'v3-mech-1'
if not TEST_MODE:
    v2sig = json.loads((V2 / 'signature.json').read_text())
    if v2sig['sim_sha256'] != SIM_SHA or v2sig['rc_sha256'] != RC_SHA:
        raise RuntimeError('Simulator or rc_prism.py differ from the files used for rc_results_v2.')
cache_dir = Path(os.environ.get('CACHE_DIR', STAGE1 / 'data_cache'))

spec = importlib.util.spec_from_file_location('prism_v3mech', sim_path)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
spec2 = importlib.util.spec_from_file_location('rc_prism_v3mech', rc_path)
rc = importlib.util.module_from_spec(spec2); sys.modules[spec2.name] = rc; spec2.loader.exec_module(rc)
rc.install(mod)

GROUPS = (np.arange(0, 4), np.arange(4, 8), np.arange(8, 10), np.arange(10, 12))  # = mod.driver_group

# ---------------- the redesigned release, installed as a patch ----------------
# MECH['payload'] = None means "control": the original simulator path, untouched.
MECH = {'payload': None, 'g_min': 1.0}
CTX = {}

def quantize(values, bits):
    """MAP-XAI quantization of a value vector over its dynamic range (post-processing)."""
    rng = max(float(np.max(np.abs(values))), 1e-8)
    levels = 2 ** bits - 1
    q = np.round((np.clip(values / rng, -1, 1) + 1.0) * 0.5 * levels) / levels
    return (2.0 * q - 1.0) * rng

def level_clip_norms(phi_cal, payload):
    """Per-endpoint clip radius for the released level, from the calibration window."""
    if payload == 'level12':
        norms = np.linalg.norm(phi_cal, axis=2)                      # (T_cal, n)
    else:
        u = np.stack([np.abs(phi_cal[:, :, g]).sum(axis=2) for g in GROUPS], axis=2)
        norms = np.linalg.norm(u, axis=2)
    c = np.quantile(norms, LEVEL_CLIP_QUANTILE, axis=0)
    pos = c[c > 1e-7]
    return np.maximum(c, max(np.median(pos) * 0.1 if len(pos) else 1e-3, 1e-5))

_privatize_orig = mod.privatize_and_compress
def privatize_v3(delta, clip_norm, sensitivity, mode, base_noise):
    if MECH['payload'] is None:
        return _privatize_orig(delta, clip_norm, sensitivity, mode, base_noise)
    B = CTX['trace'].base_gaussian
    if not (np.shares_memory(base_noise, B) and B.flags['C_CONTIGUOUS']):
        raise RuntimeError('cannot locate (t, i) of this release')
    T, n, d = B.shape
    flat = (base_noise.__array_interface__['data'][0] - B.__array_interface__['data'][0]) // (d * B.itemsize)
    t, i = divmod(int(flat), n)
    mult = sensitivity / max(clip_norm, 1e-12)                       # adjacency multiplier
    C = CTX['clip'][i]
    phi = CTX['phi'][t, i]
    k = CTX['k'][i]
    g = max(1.0 / (k + 1), MECH['g_min'])
    old = CTX['est12'][i].copy()
    if CTX['trace'].packet_failure[t, i]:
        return np.zeros(d), 0.0           # privacy is still charged by the simulator; no update
    if MECH['payload'] == 'level12':
        x = phi * min(1.0, C / max(np.linalg.norm(phi), 1e-12))
        y = x + mode.sigma * mult * C * base_noise
        kk = max(1, min(d, int(np.ceil(mode.k_fraction * d))))
        idx = np.argpartition(np.abs(y), -kk)[-kk:]
        yq = quantize(y[idx], mode.quant_bits)
        new = old.copy(); new[idx] = old[idx] + g * (yq - old[idx])   # unsent coordinates keep their value
    else:  # group4
        u = np.array([np.abs(phi[gr]).sum() for gr in GROUPS])
        u = u * min(1.0, C / max(np.linalg.norm(u), 1e-12))
        y = quantize(u + mode.sigma * mult * C * base_noise[:4], mode.quant_bits)
        CTX['est4'][i] = CTX['est4'][i] + g * (y - CTX['est4'][i])
        new = np.zeros(d)
        for gi, gr in enumerate(GROUPS):
            new[gr] = max(CTX['est4'][i][gi], 0.0) / len(gr)
    CTX['est12'][i] = new
    CTX['k'][i] += 1
    return new - old, 0.0                 # the simulator adds this to recon -> recon == est12

_attack_orig = mod.attribute_attack_auc
def attack_capture(recon_history, labels, seed):
    CTX['final_recon'] = np.array(recon_history[-1], copy=True)
    return _attack_orig(recon_history, labels, seed)

_simulate_rc = mod.simulate_scheduler
def simulate_v3(scheduler, data, outputs, trace, config, *args, **kwargs):
    n, d = outputs.phi_eval.shape[1:]
    if d != 12:
        raise RuntimeError('driver groups assume 12 features')
    CTX.update(trace=trace, phi=outputs.phi_eval, k=np.zeros(n, int),
               est12=np.zeros((n, d)), est4=np.zeros((n, 4)),
               clip=level_clip_norms(outputs.phi_cal, MECH['payload'] or 'level12'))
    res, frame = _simulate_rc(scheduler, data, outputs, trace, config, *args, **kwargs)
    if MECH['payload'] is not None and not np.allclose(CTX['final_recon'], CTX['est12'], atol=1e-9):
        raise RuntimeError('simulator reconstruction diverged from the v3 estimate')
    return res, frame

mod.privatize_and_compress = privatize_v3
mod.attribute_attack_auc = attack_capture
mod.simulate_scheduler = simulate_v3

MECH_CONFIGS = [('control', None)] + [(f'{p}_g{gm:g}', (p, gm)) for p in PAYLOADS for gm in G_MINS]
RUNS = ([('prism_xai', 'event_paced', cap) for cap in PRISM_CAPS] + [('prism_xai', 'orig', PRISM_CAPS[0])]
        + [(b, 'baseline', None) for b in ('periodic_fixed_dp', 'periodic_rr_dp', 'channel_fixed_dp',
                                            'aox_fixed_dp', 'joint_no_compression')])
DATASETS = ['synthetic'] if TEST_MODE else ['opsd', 'uci']
XAI = ['feature_ablation'] if TEST_MODE else ['treeshap', 'feature_ablation']

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
    cfg.public_release_score_threshold = PRISM_THRESHOLD
    return cfg

def run_cell(dataset, seed, xai, folder, configs):
    path = folder / 'cells' / f'{dataset}_seed{seed}_{xai}.csv'
    mark = path.with_suffix('.done.json')
    sig = {'sim': SIM_SHA, 'rc': RC_SHA, 'cell': CELL_VERSION, 'configs': [c for c, _ in configs], 'eps': EPSILONS}
    if path.is_file() and mark.is_file():
        m = json.loads(mark.read_text())
        if m.get('sha256') == sha(path) and m.get('sig') == sig:
            print('  SKIP', path.name, flush=True); return pd.read_csv(path)
    print('  RUN ', path.name, flush=True)
    cfg = base_config(dataset)
    mod.set_global_seed(seed)
    data = mod.prepare_data(cfg, seed)
    trace = mod.make_network_trace(cfg, data, seed)
    outputs = mod.train_and_explain(data, cfg, 'tree', seed, xai)
    rows = []
    for eps in EPSILONS:
        for cname, cval in configs:
            MECH['payload'], MECH['g_min'] = (None, 1.0) if cval is None else cval
            for scheduler, variant, cap in RUNS:
                rc.set_variant(variant if scheduler == 'prism_xai' else 'orig')
                cfg.max_private_releases_per_endpoint = cap if cap is not None else 8
                res, _ = mod.simulate_scheduler(scheduler, data, outputs, trace, cfg, eps, seed)
                rows.append({'dataset': dataset, 'xai_method': outputs.explainer_backend, 'seed': seed,
                             'epsilon_budget': eps, 'mechanism': cname, 'scheduler': scheduler,
                             'variant': variant, 'cap': cap,
                             'method': f'PRISM_{variant}_cap{cap}' if scheduler == 'prism_xai' else scheduler,
                             'cohen_kappa': res['driver_cohen_kappa_critical'],
                             'balanced_accuracy': res['driver_balanced_accuracy_critical'],
                             'driver_agreement': res['driver_agreement_critical'],
                             'majority_agreement_ref': res['majority_agreement_ref'],
                             'critical_coverage': res['critical_coverage'],
                             'critical_event_recall': res['critical_event_recall'],
                             'transmitted_kbits': res['transmitted_kbits'],
                             'epsilon_spent_max': res['epsilon_spent_max'],
                             'mean_aox': res['mean_aox'], 'topk_overlap': res['topk_overlap']})
    MECH['payload'], MECH['g_min'] = None, 1.0
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); df.to_csv(tmp, index=False); os.replace(tmp, path)
    mark.write_text(json.dumps({'sha256': sha(path), 'sig': sig}))
    return df

def run_grid(seeds, folder, configs):
    runs = pd.concat([run_cell(ds, s, x, folder, configs) for ds in DATASETS for s in seeds for x in XAI],
                     ignore_index=True)
    runs.to_csv(folder / 'all_runs.csv', index=False)
    return runs

# ---------------- phases ----------------
TUNE_DIR, CONF_DIR = OUT / 'tune', OUT / 'confirm'

def summarize(runs):
    return (runs.groupby(['dataset', 'mechanism', 'method'])
            [['cohen_kappa', 'balanced_accuracy', 'critical_coverage', 'critical_event_recall',
              'transmitted_kbits', 'epsilon_spent_max']].mean().reset_index())

def make_lock():
    if LOCK.exists():
        print('Lock exists; not re-selecting:', sha(LOCK)); return json.loads(LOCK.read_text())
    tune = pd.read_csv(TUNE_DIR / 'all_runs.csv')
    ep = tune[(tune.scheduler == 'prism_xai') & (tune.variant == 'event_paced')]
    m = ep.groupby(['mechanism', 'cap', 'dataset'])[['cohen_kappa', 'critical_coverage']].mean().reset_index()
    ctrl = m[m.mechanism == 'control'].set_index(['cap', 'dataset'])
    cands = []
    for (mech, cap), g in m[m.mechanism != 'control'].groupby(['mechanism', 'cap']):
        g = g.set_index('dataset')
        eligible = all(g.loc[ds, 'critical_coverage'] >= ctrl.loc[(cap, ds), 'critical_coverage'] - COV_TOL
                       for ds in g.index)
        cands.append({'mechanism': mech, 'cap': int(cap), 'eligible': bool(eligible),
                      'score': float(g.cohen_kappa.mean()),
                      **{f'kappa_{ds}': float(g.loc[ds, 'cohen_kappa']) for ds in g.index},
                      **{f'coverage_{ds}': float(g.loc[ds, 'critical_coverage']) for ds in g.index},
                      **{f'control_kappa_{ds}': float(ctrl.loc[(cap, ds), 'cohen_kappa']) for ds in g.index},
                      **{f'control_coverage_{ds}': float(ctrl.loc[(cap, ds), 'critical_coverage']) for ds in g.index}})
    cands = pd.DataFrame(cands).sort_values('score', ascending=False)
    cands.to_csv(OUT / 'lock_candidates.csv', index=False)
    ok = cands[cands.eligible]
    if ok.empty:
        chosen, go = None, False
    else:
        chosen = ok.iloc[0].to_dict()
        go = (chosen.get('kappa_opsd', chosen['score']) >= GO_KAPPA_OPSD and
              chosen.get('kappa_uci', chosen['score']) >= GO_KAPPA_UCI)
    lock = {'rule': ('PRISM event_paced only. Eligible if mean coverage on each dataset >= control coverage '
                     f'(same cap) - {COV_TOL}. Score = mean over datasets of mean dp Cohen kappa. '
                     f'GO if chosen kappa >= {GO_KAPPA_OPSD} on OPSD and >= {GO_KAPPA_UCI} on UCI.'),
            'chosen': chosen, 'go': bool(go), 'tuning_seeds': TUNE_SEEDS, 'confirm_seeds': CONFIRM_SEEDS,
            'tune_file_sha256': sha(TUNE_DIR / 'all_runs.csv'), 'sim_sha256': SIM_SHA, 'rc_sha256': RC_SHA,
            'cell_version': CELL_VERSION, 'created': dt.datetime.now().isoformat(timespec='seconds'),
            'hypotheses': ['H1 kappa(locked)-kappa(control)>0', 'H2 kappa(locked)>0',
                           f'H3 coverage diff lower 95% bound > -{NI_MARGIN}'],
            'alpha': ALPHA}
    LOCK.write_text(json.dumps(lock, indent=2, default=float))
    print('Mechanism lock SHA-256 (record it):', sha(LOCK))
    return lock

def one_sided(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2: return np.nan, np.nan, np.nan
    if np.allclose(d, d[0]):
        p = 0.0 if d[0] > 0 else 1.0
    else:
        p = stats.ttest_1samp(d, 0, alternative='greater').pvalue
    try:
        pw = stats.wilcoxon(d, alternative='greater').pvalue if np.any(d != 0) else 1.0
    except ValueError:
        pw = np.nan
    low = d.mean() - stats.t.ppf(1 - ALPHA, n - 1) * stats.sem(d) if d.std() > 0 else d.mean()
    return p, pw, low

def holm(p):
    p = np.asarray(p, float); adj = np.full_like(p, np.nan); ok = np.isfinite(p)
    idx = np.where(ok)[0]; order = idx[np.argsort(p[idx])]; run = 0.0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (len(idx) - r) * p[j])); adj[j] = run
    return adj

def confirm(lock):
    ch = lock['chosen']; mech_name = ch['mechanism']
    payload, gmin = mech_name.rsplit('_g', 1)
    configs = [('control', None), (mech_name, (payload, float(gmin)))]
    runs = run_grid(CONFIRM_SEEDS, CONF_DIR, configs)
    keys = ['dataset', 'xai_method', 'epsilon_budget', 'seed', 'method']
    a = runs[runs.mechanism == mech_name]; b = runs[runs.mechanism == 'control']
    m = a.merge(b, on=keys, suffixes=('_v3', '_ctrl'), validate='one_to_one')
    rows = []
    for (ds, xai, eps, meth), g in m.groupby(['dataset', 'xai_method', 'epsilon_budget', 'method']):
        if sorted(g.seed) != sorted(CONFIRM_SEEDS):
            raise RuntimeError(f'incomplete pairing {ds} {xai} {eps} {meth}')
        dk = g.cohen_kappa_v3 - g.cohen_kappa_ctrl
        dc = g.critical_coverage_v3 - g.critical_coverage_ctrl
        p1, w1, _ = one_sided(dk)
        p2, w2, _ = one_sided(g.cohen_kappa_v3)
        _, _, cov_low = one_sided(dc)
        rows.append({'dataset': ds, 'xai_method': xai, 'epsilon_budget': eps, 'method': meth,
                     'primary': meth == f'PRISM_event_paced_cap{int(ch["cap"])}', 'n_seeds': len(g),
                     'kappa_control': g.cohen_kappa_ctrl.mean(), 'kappa_v3': g.cohen_kappa_v3.mean(),
                     'kappa_diff': dk.mean(), 'p_H1': p1, 'p_H1_wilcoxon': w1, 'p_H2': p2,
                     'balacc_control': g.balanced_accuracy_ctrl.mean(), 'balacc_v3': g.balanced_accuracy_v3.mean(),
                     'coverage_control': g.critical_coverage_ctrl.mean(), 'coverage_v3': g.critical_coverage_v3.mean(),
                     'coverage_diff_lower95': cov_low, 'H3_noninferior': bool(cov_low > -NI_MARGIN),
                     'kbit_control': g.transmitted_kbits_ctrl.mean(), 'kbit_v3': g.transmitted_kbits_v3.mean(),
                     'eps_max_control': g.epsilon_spent_max_ctrl.mean(), 'eps_max_v3': g.epsilon_spent_max_v3.mean()})
    res = pd.DataFrame(rows)
    for fam in (True, False):                      # Holm within primary family and within secondary
        sel = res.primary == fam
        res.loc[sel, 'p_H1_holm'] = holm(res.loc[sel, 'p_H1']); res.loc[sel, 'p_H2_holm'] = holm(res.loc[sel, 'p_H2'])
    res.to_csv(OUT / 'confirmatory_v3_vs_control.csv', index=False)
    pr = res[res.primary]
    rep = ['# RC-PRISM v3 confirmatory evaluation (fresh seeds 3001-3010)', '',
           f'Mechanism lock SHA-256: {sha(LOCK)}', f'Locked: {mech_name}, PRISM event_paced cap {int(ch["cap"])}', '',
           f'H1 (kappa above control, Holm): {int((pr.p_H1_holm < ALPHA).sum())}/{len(pr)} cells',
           f'H2 (kappa above 0, Holm): {int((pr.p_H2_holm < ALPHA).sum())}/{len(pr)} cells',
           f'H3 (coverage non-inferior, margin {NI_MARGIN}): {int(pr.H3_noninferior.sum())}/{len(pr)} cells', '',
           '## Primary: PRISM event_paced', pr.round(4).to_markdown(index=False), '',
           '## Secondary: all other methods (same mechanism applied to each)',
           res[~res.primary].round(4).to_markdown(index=False)]
    write_atomic(OUT / 'REPORT_v3.md', '\n'.join(rep)); print('\n'.join(rep[:9]))

if PHASE == 'tune':
    print('== TUNE (seeds', TUNE_SEEDS, ') =='); runs = run_grid(TUNE_SEEDS, TUNE_DIR, MECH_CONFIGS)
    s = summarize(runs); s.to_csv(TUNE_DIR / 'summary.csv', index=False)
    view = s[s.method.str.startswith('PRISM_event_paced')].pivot_table(
        index=['dataset', 'method'], columns='mechanism', values='cohen_kappa')
    print('Mean dp Cohen kappa, PRISM event_paced (tuning):'); print(view.round(3).to_string())
    print('Next: set PHASE = "lock" and run again.')
elif PHASE == 'lock':
    lock = make_lock()
    print(json.dumps({k: lock[k] for k in ('chosen', 'go', 'rule')}, indent=2, default=float))
    print('GO: you may set CONFIRM_ALLOWED = True and PHASE = "confirm".' if lock['go'] else
          'NO-GO: do not run confirm; write the narrowed paper (see CLAIMS_AUDIT.md from cell 19).')
elif PHASE == 'confirm':
    lock = json.loads(LOCK.read_text())
    if not lock['go']: raise RuntimeError('Lock says NO-GO; confirmation is not allowed.')
    if not CONFIRM_ALLOWED: raise RuntimeError('Set CONFIRM_ALLOWED = True after reading the lock.')
    if lock['sim_sha256'] != SIM_SHA or lock['rc_sha256'] != RC_SHA or lock['cell_version'] != CELL_VERSION:
        raise RuntimeError('Code changed since the lock was written.')
    confirm(lock)
print('RESULTS:', OUT)
