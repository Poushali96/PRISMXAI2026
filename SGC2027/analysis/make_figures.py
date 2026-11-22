"""Builds every number (results_macros.tex) and every result figure of the SmartGridComm paper
directly from the result files. Nothing is typed by hand.

Inputs (../data):
  C_EXPERIMENTAL_PUBLIC_SCORE_M8_*        720-run matrix (PRISM vs MAP-XAI-only, external baselines)
  heldout_all_run_metrics.csv             v2 confirmatory runs, fresh seeds 2001-2010
  confirmatory_ablation_vs_orig.csv       v2 pacing tests (seeds as units, Holm)
  claims_with_baseline_coverage.csv       v3 claims audit
  driver_diagnosis_runs.csv               v3 diagnosis (tuning seeds 11,22,33; dp vs no-noise)
  rc_results_v3_mech/                     v3 pre-registered redesign (tuning) + lock
  ../analysis/prop_validation.csv         mechanism-level check of Proposition 1
"""
import json, hashlib, math, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

D = Path('/home/claude/w/data'); A = Path('/home/claude/w/analysis')
FIG = Path('/home/claude/w/paper/figures'); FIG.mkdir(parents=True, exist_ok=True)
MAC = {}

# ---------------- style (IEEE column = 3.5 in, Times-like STIX text) ----------------
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5,
    'axes.labelsize': 7.5, 'axes.titlesize': 8, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
    'ytick.color': '#52514e', 'grid.color': '#e6e5e0', 'grid.linewidth': 0.5, 'axes.grid': True,
    'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 300,
    'axes.titleweight': 'bold', 'axes.titlepad': 3,
})
# validated categorical palette, fixed order, colour follows the entity
C1, C2, C3, C4, C5, C6, C7, C8 = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
                                  '#6250d6', '#e34948')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#9a9893'
COL = {'PRISM_event_paced': C1, 'PRISM_orig': C2, 'PRISM_paced': C3, 'aox_fixed_dp': C4,
       'periodic_rr_dp': C5, 'periodic_fixed_dp': C6, 'channel_fixed_dp': C7, 'joint_no_compression': C8}
NAME = {'PRISM_event_paced': 'PRISM event-paced', 'PRISM_orig': 'PRISM original', 'PRISM_paced': 'PRISM paced',
        'aox_fixed_dp': 'AoX fixed-DP', 'periodic_rr_dp': 'Round-robin DP', 'periodic_fixed_dp': 'Periodic fixed-DP',
        'channel_fixed_dp': 'Channel fixed-DP', 'joint_no_compression': 'Joint full-packet'}

def mac(name, value, fmt='{}'):
    MAC[name] = fmt.format(value)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# =====================================================================================
# 1. Matched component result (720-run matrix): PRISM vs MAP-XAI-only
# =====================================================================================
pv = pd.read_csv(D / 'C_EXPERIMENTAL_PUBLIC_SCORE_M8_paired_prism_vs_map.csv').set_index('metric')
mm = pd.read_csv(D / 'C_EXPERIMENTAL_PUBLIC_SCORE_M8_method_means.csv').set_index('scheduler')
mac('mapTrafficDiff', -pv.loc['transmitted_kbits', 'difference_mean'], '{:.2f}')
mac('mapTrafficCI', pv.loc['transmitted_kbits', 'ci95_half_width'], '{:.2f}')
mac('mapTrafficRel', -100 * pv.loc['transmitted_kbits', 'relative_difference_mean'], '{:.2f}')
mac('mapPairs', int(pv.loc['transmitted_kbits', 'n_pairs']))
mac('mapPrismKbit', mm.loc['prism_xai', 'transmitted_kbits'], '{:.1f}')
mac('mapMapKbit', mm.loc['repaired_mapxai', 'transmitted_kbits'], '{:.1f}')
mac('mapRecallDiffPP', 100 * pv.loc['critical_event_recall', 'difference_mean'], '{:+.2f}')
mac('mapRecallQ', pv.loc['critical_event_recall', 'q_bh_t'], '{:.2f}')
mac('mapCovDiffPP', 100 * pv.loc['intervention_coverage', 'difference_mean'], '{:+.2f}')
q = pv.loc['transmitted_kbits', 'q_bh_t']
mac('mapTrafficQexp', int(math.floor(math.log10(q))) + 1)
ext = ['periodic_fixed_dp', 'channel_fixed_dp', 'aox_fixed_dp', 'joint_no_compression']
red = [100 * (1 - mm.loc['repaired_mapxai', 'transmitted_kbits'] / mm.loc[b, 'transmitted_kbits']) for b in ext]
mac('mapExtRedMin', min(red), '{:.1f}'); mac('mapExtRedMax', max(red), '{:.1f}')
TABLE_MAIN = mm.loc[ext + ['repaired_mapxai', 'prism_xai']]

# =====================================================================================
# 2. Pacing: confirmatory tests on fresh seeds (v2)
# =====================================================================================
ab = pd.read_csv(D / 'confirmatory_ablation_vs_orig.csv')
for ds in ['opsd', 'uci']:
    for v in ['event_paced', 'paced']:
        g = ab[(ab.dataset == ds) & (ab.variant == v)]
        tag = ds.upper() + ('Ev' if v == 'event_paced' else 'Pa')
        mac(f'pace{tag}Up', int(((g.p_coverage_holm < .05) & (g.coverage_diff > 0)).sum()))
        mac(f'pace{tag}Down', int(((g.p_coverage_holm < .05) & (g.coverage_diff < 0)).sum()))
        mac(f'pace{tag}N', len(g))
        mac(f'pace{tag}Med', 100 * g.coverage_diff.median(), '{:+.1f}')
        mac(f'pace{tag}CovO', 100 * g.coverage_orig.mean(), '{:.1f}')
        mac(f'pace{tag}CovV', 100 * g.coverage_variant.mean(), '{:.1f}')
        mac(f'pace{tag}AoxO', g.mean_aox_orig.mean(), '{:.0f}')
        mac(f'pace{tag}AoxV', g.mean_aox_variant.mean(), '{:.0f}')
        mac(f'pace{tag}Kbit', g.kbit_diff.abs().max(), '{:.1f}')
        mac(f'pace{tag}KbitRel', 100 * (g.kbit_diff.abs() / g.kbit_orig).max(), '{:.1f}')
        mac(f'pace{tag}Eps', g.eps_max_diff.abs().max(), '{:.2f}')
        mac(f'pace{tag}RecUp', int(((g.p_recall_holm < .05) & (g.recall_diff > 0)).sum()))
mac('paceNtests', len(ab))
ev = ab[ab.variant == 'event_paced']; rel = ev.kbit_diff / ev.kbit_orig
mac('paceKbitUpMax', 100 * rel.max(), '{:.1f}'); mac('paceKbitDownMax', -100 * rel.min(), '{:.1f}')
mac('paceKbitMedOPSD', 100 * rel[ev.dataset == 'opsd'].median(), '{:+.1f}'); mac('paceKbitMedUCI', 100 * rel[ev.dataset == 'uci'].median(), '{:+.1f}')
mac('paceEpsUpMax', ev.eps_max_diff.max(), '{:.2f}'); mac('paceEpsDownMax', -ev.eps_max_diff.min(), '{:.2f}')
held = pd.read_csv(D / 'heldout_all_run_metrics.csv', low_memory=False)
mac('heldRows', len(held))

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.78), sharex=False, sharey=False)
for ax, ds in zip(axes, ['opsd', 'uci']):
    g = ab[ab.dataset == ds]
    lim = max(g.coverage_orig.max(), g.coverage_variant.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.fill_between([0, lim], [0, lim], [lim, lim], color='#2a78d6', alpha=0.045, lw=0, zorder=0)
    for v, c, mk in [('paced', C3, 's'), ('event_paced', C1, 'o')]:
        s = g[g.variant == v]
        sig_up = (s.p_coverage_holm < .05) & (s.coverage_diff > 0)
        sig_dn = (s.p_coverage_holm < .05) & (s.coverage_diff < 0)
        ns = ~(sig_up | sig_dn)
        ax.scatter(s.coverage_orig[sig_up] * 100, s.coverage_variant[sig_up] * 100, s=11, marker=mk, color=c,
                   edgecolor='white', linewidth=0.4, zorder=3)
        ax.scatter(s.coverage_orig[ns] * 100, s.coverage_variant[ns] * 100, s=11, marker=mk, facecolor='white',
                   edgecolor=c, linewidth=0.7, zorder=3)
        ax.scatter(s.coverage_orig[sig_dn] * 100, s.coverage_variant[sig_dn] * 100, s=14, marker='x', color=c,
                   linewidth=0.8, zorder=3)
    ax.set_xlim(0, lim * 100); ax.set_ylim(0, lim * 100)
    ax.set_title(ds.upper(), loc='left')
    ax.set_xlabel('Coverage, original PRISM (%)')
    up = int(((g.variant == 'event_paced') & (g.p_coverage_holm < .05) & (g.coverage_diff > 0)).sum())
    ax.text(0.97, 0.05, f'event-paced: {up}/{(g.variant=="event_paced").sum()}\nsignificantly higher',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.2, color=INK2)
axes[0].set_ylabel('Coverage, paced variant (%)')
h1 = [Line2D([], [], marker='o', ls='', color=C1, markersize=3.6, label='event-paced'),
      Line2D([], [], marker='s', ls='', color=C3, markersize=3.4, label='paced'),
      Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=INK2, markersize=3.4,
             label='n.s. (Holm)'),
      Line2D([], [], marker='x', ls='', color=INK2, markersize=3.6, label='lower (Holm)')]
fig.legend(handles=h1, loc='upper center', ncol=4, bbox_to_anchor=(0.54, 1.03), handletextpad=0.2,
           columnspacing=0.9)
fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.92))
fig.savefig(FIG / 'fig_pacing.pdf'); plt.close(fig)

# =====================================================================================
# 3. Operating points on held-out seeds (traffic vs coverage)
# =====================================================================================
held['method'] = np.where(held.setting_kind == 'PRISM', 'PRISM_' + held.variant.astype(str), held.scheduler)
FM = pd.read_csv(A / 'frontier_margins_heldout.csv')
for ds in ['opsd', 'uci']:
    for m, tg in [('PRISM_event_paced', 'Ev'), ('PRISM_orig', 'Or'), ('PRISM_paced', 'Pa')]:
        fm = FM[(FM.dataset == ds) & (FM.method == m)]
        mac(f'fr{tg}{ds.upper()}Above', 100 * (fm.margin > 0).mean(), '{:.1f}')
        mac(f'fr{tg}{ds.upper()}Med', 100 * fm.margin.median(), '{:+.1f}')
        mac(f'fr{tg}{ds.upper()}N', len(fm))
    mac(f'fr{ds.upper()}KbitMax', FM[FM.dataset == ds].kbit.max(), '{:.0f}')
cm = FM[FM.method == 'PRISM_event_paced'].groupby(['dataset', 'xai', 'eps', 'setting']).margin.mean()
mac('frCellsPos', int((cm > 0).sum())); mac('frCells', len(cm))
pts = held.groupby(['dataset', 'method', 'setting'])[['transmitted_kbits', 'critical_coverage']].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
order_b = ['aox_fixed_dp', 'periodic_rr_dp', 'periodic_fixed_dp', 'channel_fixed_dp', 'joint_no_compression']
for ax, ds in zip(axes, ['opsd', 'uci']):
    p = pts[pts.dataset == ds]
    hb = held[(held.dataset == ds) & (held.setting_kind == 'BASELINE')]
    for b in order_b:
        q = hb[hb.scheduler == b].groupby('baseline_duty_cycle')[['transmitted_kbits', 'critical_coverage']].mean()
        q = q.sort_values('transmitted_kbits')
        ax.plot(q.transmitted_kbits, q.critical_coverage * 100, color=COL[b], lw=1.1, marker='o', markersize=2.4,
                markeredgecolor='white', markeredgewidth=0.3, zorder=2)
    for m, mk in [('PRISM_orig', 'v'), ('PRISM_paced', 's'), ('PRISM_event_paced', 'o')]:
        q = p[p.method == m]
        ax.scatter(q.transmitted_kbits, q.critical_coverage * 100, s=9, marker=mk, color=COL[m],
                   edgecolor='white', linewidth=0.3, zorder=3, alpha=0.95)
    ax.set_xscale('log'); ax.set_title(ds.upper(), loc='left')
    fm = FM[(FM.dataset == ds) & (FM.method == 'PRISM_event_paced')]
    ax.text(0.03, 0.97, f'event-paced above\nbaseline envelope:\n{100*(fm.margin>0).mean():.0f}% of runs,\nmedian {100*fm.margin.median():+.1f} pp',
            transform=ax.transAxes, ha='left', va='top', fontsize=5.9, color=INK2, linespacing=1.15)
    ax.set_xlabel('Transmitted traffic (kbit, log)'); ax.set_ylim(-3, 103)
    ax.grid(True, which='minor', color='#f1f0ec', lw=0.35)
axes[0].set_ylabel('Critical coverage (%)')
hs = [Line2D([], [], color=COL[m], marker=mk, ls='', markersize=3.2, label=NAME[m])
      for m, mk in [('PRISM_event_paced', 'o'), ('PRISM_paced', 's'), ('PRISM_orig', 'v')]] + \
     [Line2D([], [], color=COL[b], lw=1.1, label=NAME[b]) for b in order_b]
fig.legend(handles=hs, loc='upper center', ncol=3, bbox_to_anchor=(0.53, 1.04), handlelength=1.3,
           handletextpad=0.3, columnspacing=0.7)
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.80))
fig.savefig(FIG / 'fig_frontier.pdf'); plt.close(fig)

for ds in ['opsd', 'uci']:
    hb = held[(held.dataset == ds) & (held.setting_kind == 'BASELINE') & (held.baseline_duty_cycle == 1.0)]
    g = hb.groupby('scheduler')[['transmitted_kbits', 'critical_coverage']].mean()
    mac(f'opAox{ds.upper()}Cov', 100 * g.loc['aox_fixed_dp', 'critical_coverage'], '{:.0f}')
    mac(f'opRr{ds.upper()}Cov', 100 * g.loc['periodic_rr_dp', 'critical_coverage'], '{:.0f}')
    mac(f'opAox{ds.upper()}Kbit', g.loc['aox_fixed_dp', 'transmitted_kbits'], '{:.0f}')
    pe = pts[(pts.dataset == ds) & (pts.method == 'PRISM_event_paced')]
    mac(f'opEv{ds.upper()}CovMax', 100 * pe.critical_coverage.max(), '{:.0f}')
    mac(f'opEv{ds.upper()}KbitMax', pe.transmitted_kbits.max(), '{:.0f}')
    mac(f'opEv{ds.upper()}KbitMin', pe.transmitted_kbits.min(), '{:.0f}')

# =====================================================================================
# 4. Claims audit (v3, held-out v2)
# =====================================================================================
cl = pd.read_csv(D / 'claims_with_baseline_coverage.csv')
t = cl[cl.status == 'tested']
mac('claimTested', len(t)); mac('claimSupported', int(t.claim_supported.sum())); mac('claimAll', len(cl))
mac('claimUntested', len(cl) - len(t))
tier = t.baseline_tier.str.split(' ').str[0]
for k in ['degenerate', 'weak', 'strong']:
    mac(f'claim{k.capitalize()}N', int((tier == k).sum()))
    mac(f'claim{k.capitalize()}S', int(t[tier == k].claim_supported.sum()))
TIER = (t.assign(tier=tier).groupby(['dataset', 'tier'])
        .agg(tested=('claim_supported', 'size'), supported=('claim_supported', 'sum'),
             red=('relative_reduction_pct', 'mean'), bcov=('baseline_coverage', 'mean')).reset_index())

# =====================================================================================
# 5. Fidelity diagnosis (v3; tuning seeds) + redesign + Proposition check
# =====================================================================================
dg = pd.read_csv(D / 'driver_diagnosis_runs.csv')
dg['method'] = np.where(dg.scheduler == 'prism_xai', 'PRISM_' + dg.variant, dg.scheduler)
S = dg.pivot_table(index=['dataset', 'method'], columns='noise',
                   values=['cohen_kappa', 'recon_norm_ratio_end', 'balanced_accuracy'], aggfunc='mean')
S.columns = [f'{a}_{b}' for a, b in S.columns]; S = S.reset_index()
bl = ['aox_fixed_dp', 'periodic_rr_dp', 'periodic_fixed_dp', 'channel_fixed_dp']
pr = ['PRISM_orig', 'PRISM_paced', 'PRISM_event_paced']
so = S[(S.dataset == 'opsd') & S.method.isin(bl)]
mac('diagBlKappaNNmin', so.cohen_kappa_no_noise.min(), '{:.2f}'); mac('diagBlKappaNNmax', so.cohen_kappa_no_noise.max(), '{:.2f}')
mac('diagBlKappaDPmax', so.cohen_kappa_dp.abs().max(), '{:.3f}')
mac('diagBlRatioMin', so.recon_norm_ratio_end_dp.min(), '{:.0f}'); mac('diagBlRatioMax', so.recon_norm_ratio_end_dp.max(), '{:.0f}')
mac('diagBlRatioNNmax', so.recon_norm_ratio_end_no_noise.max(), '{:.2f}')
su = S[(S.dataset == 'uci') & S.method.isin(bl)]
mac('diagUciRatioMin', su.recon_norm_ratio_end_dp.min(), '{:.0f}'); mac('diagUciRatioMax', su.recon_norm_ratio_end_dp.max(), '{:.0f}')
mac('diagUciKappaNNmax', su.cohen_kappa_no_noise.max(), '{:.2f}')
sp = S[S.method.isin(pr)]
mac('diagPrKappaNNmin', sp.cohen_kappa_no_noise.min(), '{:.3f}'); mac('diagPrKappaNNmax', sp.cohen_kappa_no_noise.max(), '{:.3f}')
mac('diagSeeds', ', '.join(str(s) for s in sorted(dg.seed.unique())))
mac('diagRuns', len(dg))

mech = pd.read_csv(D / 'rc_results_v3_mech/tune/summary.csv')
lock = json.loads((D / 'rc_results_v3_mech/mechanism_lock.json').read_text())
mac('lockSha', sha(D / 'rc_results_v3_mech/mechanism_lock.json')[:12])
mac('lockVtwoSha', sha(D / 'selection_lock_v2.json')[:12])
ch = lock['chosen']
mac('redKappaOpsd', ch['kappa_opsd'], '{:+.3f}'); mac('redKappaUci', ch['kappa_uci'], '{:+.3f}')
mac('redMech', ch['mechanism'].replace('_', r'\_'))
evm = mech[mech.method.str.startswith('PRISM_event_paced') & (mech.mechanism != 'control')]
mac('redAbsMax', evm.cohen_kappa.abs().max(), '{:.3f}')
allm = mech[mech.mechanism != 'control']
mac('redAllAbsMax', allm.cohen_kappa.abs().max(), '{:.3f}')
mac('redNconfigs', mech.mechanism.nunique() - 1)

pvd = pd.read_csv(A / 'prop_validation.csv')
fin = pvd[pvd.k == pvd.k.max()]
errs = (fin.err_feedback_rms - fin.theory_rms).abs() / fin.theory_rms
mac('propMaxRelErr', 100 * errs.max(), '{:.1f}')
mac('propK', int(pvd.k.max()))
g8 = fin[(fin.sigma == 8) & (fin['mode'].str.startswith('MAP'))].iloc[0]
mac('propEFeight', g8.err_feedback_rms, '{:.0f}'); mac('propLvEight', g8.level_avg, '{:.2f}')

# PRISM per-release noise multiplier (budget-calibrated) for context
spec = importlib.util.spec_from_file_location('sim', '/home/claude/drive/sim.py')
sim = importlib.util.module_from_spec(spec); sys.modules['sim'] = sim; spec.loader.exec_module(sim)
cfg = sim.ExperimentConfig.full_defaults()
sig = {}
for eps in [8, 16, 32]:
    acc = sim.ZCDPAccountant(eps, cfg.delta) if 'ZCDPAccountant' in dir(sim) else None
    for cap in [8, 32]:
        cfg.max_private_releases_per_endpoint = cap
        sig[(eps, cap)] = sim.budget_calibrated_sigma(acc, cfg)
mac('sigPrismLo', min(sig.values()), '{:.2f}'); mac('sigPrismHi', max(sig.values()), '{:.2f}')
mac('sigBase', sim.fixed_mode().sigma, '{:.0f}')

fig = plt.figure(figsize=(7.16, 2.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.12, 1.0, 1.15, 1.0], wspace=0.48)
# (a) dumbbell kappa no-noise -> dp
ax = fig.add_subplot(gs[0])
meths = bl + ['joint_no_compression'] + pr[::-1]
y0 = np.arange(len(meths))
for j, (ds, off, mk) in enumerate([('opsd', -0.17, 'o'), ('uci', 0.17, 's')]):
    s = S[S.dataset == ds].set_index('method').loc[meths]
    for i, m in enumerate(meths):
        ax.plot([s.cohen_kappa_no_noise[m], s.cohen_kappa_dp[m]], [i + off] * 2, color=COL[m], lw=0.9, alpha=0.55)
        ax.scatter(s.cohen_kappa_no_noise[m], i + off, s=13, marker=mk, facecolor='white', edgecolor=COL[m],
                   linewidth=0.8, zorder=3)
        ax.scatter(s.cohen_kappa_dp[m], i + off, s=13, marker=mk, color=COL[m], edgecolor='white',
                   linewidth=0.3, zorder=4)
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_yticks(y0); ax.set_yticklabels([NAME[m] for m in meths], fontsize=6.2)
ax.set_xlabel(r"Cohen's $\kappa$ (driver group)"); ax.grid(axis='y', visible=False)
ax.set_title('(a) Driver fidelity', loc='left')
ax.legend(handles=[Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=INK2, markersize=3.4, label='no noise'),
                   Line2D([], [], marker='o', ls='', color=INK2, markersize=3.4, label='with DP'),
                   Line2D([], [], marker='o', ls='', color=MUTED, markersize=3, label='OPSD'),
                   Line2D([], [], marker='s', ls='', color=MUTED, markersize=3, label='UCI')],
          loc='upper right', fontsize=5.6, handletextpad=0.1, borderaxespad=0.0,
          ncol=1, labelspacing=0.2)
ax.set_xlabel(r"Cohen's $\kappa$ (driver group)", labelpad=1)
# (b) reconstruction norm ratio
ax = fig.add_subplot(gs[1])
mb = bl + ['joint_no_compression'] + pr
x = np.arange(len(mb)); w = 0.38
for j, (ds, off, hatch) in enumerate([('opsd', -w / 2, None), ('uci', w / 2, '////')]):
    s = S[S.dataset == ds].set_index('method').loc[mb]
    ax.bar(x + off, s.recon_norm_ratio_end_dp, width=w - 0.04, color=[COL[m] for m in mb], edgecolor='white',
           linewidth=0.4, hatch=hatch, zorder=2)
    ax.scatter(x + off, s.recon_norm_ratio_end_no_noise, marker='_', color=INK, s=22, linewidth=1.0, zorder=3)
ax.set_yscale('log'); ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(x); ax.set_xticklabels(['AoX', 'RR', 'Per.', 'Chan.', 'Joint', 'P-orig', 'P-pace', 'P-ev'],
                                     rotation=55, ha='right', fontsize=5.9)
ax.set_ylabel(r'$\|\hat{\phi}_z\| / \|\phi_z\|$ (end)'); ax.grid(axis='x', visible=False)
ax.set_title('(b) Reconstruction drift', loc='left')
ax.legend(handles=[Patch(facecolor=MUTED, label='OPSD, DP'), Patch(facecolor='white', edgecolor=MUTED, hatch='////', label='UCI, DP'),
                   Line2D([], [], marker='_', ls='', color=INK, markersize=6, label='no noise')],
          loc='upper right', fontsize=5.6, borderaxespad=0.1, handlelength=1.1, labelspacing=0.25)
# (c) Proposition 1: theory vs simulator release function
ax = fig.add_subplot(gs[2])
for s_, c_ in [(2.0, C3), (8.0, C1), (12.0, C7)]:
    q = pvd[(pvd.sigma == s_) & pvd['mode'].str.startswith('MAP')]
    ax.plot(q.k, q.theory_rms, color=c_, lw=1.2, zorder=2)
    qq = q[q.k % 25 == 0]
    ax.scatter(qq.k, qq.err_feedback_rms, s=8, color=c_, edgecolor='white', linewidth=0.3, zorder=3)
    ax.plot(q.k, q.level_avg, color=c_, lw=0.9, ls=(0, (2, 1.5)), zorder=2)
    ax.text(q.k.max() * 1.02, q.theory_rms.iloc[-1], rf'$\sigma$={s_:g}', fontsize=6, color=INK2, va='center')
ax.set_yscale('log'); ax.set_xlim(0, pvd.k.max() * 1.18)
ax.set_xlabel('Delivered releases $k$'); ax.set_ylabel(r'RMS error $\|e_k\|/C$')
ax.set_title('(c) Proposition 1 check', loc='left')
ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, label='theory (error feedback)'),
                   Line2D([], [], color=INK2, marker='o', ls='', markersize=2.8, label='simulated'),
                   Line2D([], [], color=INK2, lw=0.9, ls=(0, (2, 1.5)), label='level + running mean')],
          loc='center right', bbox_to_anchor=(1.0, 0.42), fontsize=5.6, borderaxespad=0.1, handlelength=1.6, labelspacing=0.25)
# (d) pre-registered redesign: PRISM event-paced kappa heatmap
ax = fig.add_subplot(gs[3])
cfgs = ['control'] + [f'{p}_g{g}' for p in ['level12', 'group4'] for g in ['1', '0.5', '0.25', '0.1']]
cols = [('opsd', 'PRISM_event_paced_cap8'), ('opsd', 'PRISM_event_paced_cap32'),
        ('uci', 'PRISM_event_paced_cap8'), ('uci', 'PRISM_event_paced_cap32')]
Mx = np.array([[mech[(mech.dataset == d_) & (mech.method == m_) & (mech.mechanism == c_)].cohen_kappa.iloc[0]
                for d_, m_ in cols] for c_ in cfgs])
vmax = 0.10
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('div', ['#d95926', '#f4f3ef', '#2a78d6'])
im = ax.imshow(Mx, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        ax.text(j, i, f'{Mx[i, j]:+.2f}'.replace('+0.', '+.').replace('-0.', '−.'), ha='center', va='center',
                fontsize=5.2, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(['OPSD\n$M$=8', 'OPSD\n$M$=32', 'UCI\n$M$=8', 'UCI\n$M$=32'], fontsize=5.8)
lab = ['control'] + [rf'{p} $g$={g}' for p in ['L12', 'G4'] for g in ['1', '.5', '.25', '.1']]
ax.set_yticks(range(len(cfgs))); ax.set_yticklabels(lab, fontsize=5.6); ax.grid(False)
ax.tick_params(length=0)
for sp_ in ax.spines.values(): sp_.set_visible(False)
ax.set_title(r'(d) Redesign, PRISM $\kappa$', loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, ticks=[-0.1, 0, 0.1]); cb.ax.tick_params(labelsize=5.6, length=1.5)
cb.ax.set_yticklabels(['−.10', '0', '+.10'])
cb.outline.set_visible(False)
fig.subplots_adjust(left=0.115, right=0.955, bottom=0.27, top=0.88)
fig.savefig(FIG / 'fig_fidelity.pdf'); plt.close(fig)

# =====================================================================================
# write macros + tables
# =====================================================================================
lines = ['% Auto-generated by analysis/make_figures.py from the result files. Do not edit by hand.']
for k, v in MAC.items():
    lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
(Path('/home/claude/w/paper') / 'results_macros.tex').write_text('\n'.join(lines) + '\n')

def r(x, f): return f.format(x)
rows = []
lab = {'periodic_fixed_dp': 'Periodic fixed-DP', 'channel_fixed_dp': 'Channel fixed-DP', 'aox_fixed_dp': 'AoX fixed-DP',
       'joint_no_compression': 'Joint full-packet', 'repaired_mapxai': r'\mapxai{}-only (ours)', 'prism_xai': r'\method{} (ours)'}
for k_, row in TABLE_MAIN.iterrows():
    kb = r(row.transmitted_kbits, '{:.1f}')
    if k_ == 'prism_xai': kb = r'\textbf{' + kb + '}'
    rows.append(f"{lab[k_]} & {kb} & {row.intervention_coverage:.3f} & {row.critical_event_recall:.3f} & {row.epsilon_spent_max:.2f}\\\\")
    if k_ == 'joint_no_compression': rows.append(r'\midrule')
(Path('/home/claude/w/paper') / 'table_main_rows.tex').write_text('\n'.join(rows) + '\n')

trow = []
nm = {'degenerate': 'Degenerate ($<$5\\%)', 'weak': 'Weak (5--50\\%)', 'strong': 'Strong ($\\geq$50\\%)'}
for _, rr in TIER.iterrows():
    trow.append(f"{rr.dataset.upper()} & {nm[rr.tier]} & {int(rr.supported)}/{int(rr.tested)} & {rr.red:.1f} & {100*rr.bcov:.1f}\\\\")
(Path('/home/claude/w/paper') / 'table_tier_rows.tex').write_text('\n'.join(trow) + '\n')
print(json.dumps(MAC, indent=0)[:4000])
print('sigma table', {f'{k}': round(v, 2) for k, v in sig.items()})
