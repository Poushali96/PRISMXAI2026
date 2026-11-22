"""Mechanism-level validation of Proposition 1 using the simulator's own release function.
No dataset is involved: a fixed attribution vector phi (||phi|| = L*C) is streamed through
(a) the clipped error-feedback update used by the simulator's DP baselines, and
(b) a level release with running average (the pre-registered redesign, g_min -> 1/k).
Outputs prop_validation.csv."""
import sys, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('sim', '/home/claude/drive/sim.py')
sim = importlib.util.module_from_spec(spec); sys.modules['sim'] = sim; spec.loader.exec_module(sim)
d, C, L, K, R = 12, 1.0, 1.0, 400, 300
rng = np.random.default_rng(7)
phi0 = rng.normal(size=d); phi0 *= L * C / np.linalg.norm(phi0)
modes = {'gaussian (k=d, 16-bit)': sim.PacketMode('g', 1.0, 16, None)}
rows = []
def theory(sig, K):
    s2 = (sig * C) ** 2; m = np.zeros(K + 1); m[0] = L * C   # e_0 = -phi
    for k in range(1, K + 1):
        e = m[k - 1]
        m[k] = np.sqrt(max(e - C, 0) ** 2 + d * s2) if e > C else np.sqrt(d * s2)
    return m
for sig in [2.0, 8.0, 12.0]:
    for mname, mode in [('gaussian', sim.PacketMode('g', 1.0, 16, sig)),
                        ('MAP-XAI balanced (k=d/2, 8-bit)', sim.PacketMode('b', 0.5, 8, sig))]:
        E_ef = np.zeros((R, K + 1)); E_lv = np.zeros((R, K + 1))
        for r in range(R):
            rec = np.zeros(d); est = np.zeros(d)
            E_ef[r, 0] = E_lv[r, 0] = np.linalg.norm(phi0)
            for k in range(1, K + 1):
                z = rng.normal(size=d)
                dec, _ = sim.privatize_and_compress(phi0 - rec, C, C, mode, z)   # error feedback
                rec = rec + dec
                lv, _ = sim.privatize_and_compress(phi0, C * 1.0, C, mode, z)   # level, ||phi||<=C
                est = est + (lv - est) / k
                E_ef[r, k] = np.linalg.norm(rec - phi0); E_lv[r, k] = np.linalg.norm(est - phi0)
        th = theory(sig, K)
        for k in range(K + 1):
            rows.append(dict(sigma=sig, mode=mname, k=k, err_feedback=E_ef[:, k].mean(),
                             err_feedback_rms=np.sqrt((E_ef[:, k]**2).mean()),
                             level_avg=E_lv[:, k].mean(), theory_rms=th[k],
                             steady_state=(d * (sig*C)**2 + C**2) / (2*C)))
        print(sig, mname, 'EF rms@K', np.sqrt((E_ef[:, K]**2).mean()).round(1), 'theory', th[K].round(1),
              'steady', ((d*(sig*C)**2 + C**2)/(2*C)), 'level@K', E_lv[:, K].mean().round(2))
pd.DataFrame(rows).to_csv('prop_validation.csv', index=False)
