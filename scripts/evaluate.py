"""Main experiments: training curve, port/pilot/user/AP/length sweeps, per-user CDF, runtime and candidate-size ablation."""
import numpy as np, json, time
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flarefas.system import *
from flarefas import model as flare

ap = argparse.ArgumentParser(description='Reproduce Figs. 3-9 and Tables 2-4 of the paper.')
ap.add_argument('--weights', default=str(ROOT / 'weights' / 'flare_params.npz'))
ap.add_argument('--realizations', type=int, default=240, help='network realizations per point (multiple of 60)')
ap.add_argument('--quick', action='store_true', help='60 realizations per point for a fast smoke run')
args = ap.parse_args()
FIG = ROOT / 'figures'; RES = ROOT / 'results'; FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
P, hist = flare.load_params(args.weights)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 8, 'axes.grid': True, 'grid.alpha': .3, 'lines.linewidth': 1.0, 'axes.linewidth': 0.6})
LS = {'Fixed port': ':', 'Random port': (0, (1, 2)), 'Max-gain': '-.', 'Greedy-CD': '-', 'FLARE': '-', 'FLARE-LS': '--', 'Greedy-CD (perfect CSI)': (0, (5, 2))}
MK = {'Fixed port': ('s', 'tab:gray'), 'Random port': ('v', 'tab:brown'), 'Max-gain': ('^', 'tab:green'),
      'Greedy-CD': ('o', 'tab:blue'), 'FLARE': ('*', 'tab:red'), 'FLARE-LS': ('D', 'tab:purple'),
      'Greedy-CD (perfect CSI)': ('x', 'k')}
NS = 60 if args.quick else args.realizations   # realizations per point (batches of 60)


def run_methods(rng, L, K, N, W=2.0, pdbm=20.0, genie=False, per_user=False):
    out = {}; tims = {}
    for b in range(NS//60):
        H, Hh, _ = gen_batch(rng, 60, L, K, N, W=W, pilot_dbm=pdbm)
        meth = {'Fixed port': lambda: sel_fixed(Hh), 'Random port': lambda: sel_random(Hh, rng),
                'Max-gain': lambda: sel_maxgain(Hh), 'Greedy-CD': lambda: sel_greedy(Hh),
                'FLARE': lambda: flare.sel_flare(P, Hh), 'FLARE-LS': lambda: flare.sel_flare_ls(P, Hh)}
        for m, fn in meth.items():
            t = time.time(); p = fn(); tims.setdefault(m, []).append((time.time()-t)/60)
            r = rates(select(H, p), rzf(select(Hh, p)))
            out.setdefault(m, []).append(r if per_user else r.sum(1))
        if genie:
            p = sel_greedy(H)
            out.setdefault('Greedy-CD (perfect CSI)', []).append(rates(select(H, p), rzf(select(H, p))).sum(1))
    out = {m: np.concatenate(v) for m, v in out.items()}
    return out, {m: float(np.mean(v)) for m, v in tims.items()}


def sweep(name, xs, kw_fn, xlabel, genie=False, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for x in xs:
        o, _ = run_methods(rng, **kw_fn(x), genie=genie)
        for m, v in o.items():
            res.setdefault(m, []).append(float(v.mean()))
        print(name, x, {m: round(v[-1], 2) for m, v in res.items()})
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m, v in res.items():
        mk, c = MK[m]
        ax.plot(xs, v, marker=mk, color=c, label=m, ls=LS[m], ms=4, mfc='none' if m in ('Greedy-CD','FLARE-LS') else c)
    ax.set_xlabel(xlabel); ax.set_ylabel('Average sum SE (bit/s/Hz)')
    ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig(FIG / f'fig_{name}.png', dpi=600); plt.close(fig)
    return res


R = {}
# Fig: training curve
h = np.array(hist)
fig, ax = plt.subplots(figsize=(3.3, 2.5)); ax2 = ax.twinx()
ax.plot(h[:, 0]+1, h[:, 1], 'b-o', ms=3, label='Train CE'); ax.plot(h[:, 0]+1, h[:, 2], 'g-s', ms=3, label='Val. CE')
ax2.plot(h[:, 0]+1, h[:, 3]*100, 'r-^', ms=3, label='Val. top-1 acc.')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy'); ax2.set_ylabel('Top-1 port accuracy (%)'); ax2.grid(False)
l1, a1 = ax.get_legend_handles_labels(); l2, a2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, a1+a2, fontsize=6.5, loc='center right'); fig.tight_layout(); fig.savefig(FIG / 'fig_training.png', dpi=600); plt.close(fig)
R['hist'] = hist

R['ports'] = sweep('ports', [4, 8, 16, 32], lambda x: dict(L=16, K=6, N=x), 'Number of fluid-antenna ports N')
R['pilot'] = sweep('pilot', [0, 10, 20, 30, 40], lambda x: dict(L=16, K=6, N=16, pdbm=float(x)),
                   'Effective pilot power (dBm)', genie=True)
R['users'] = sweep('users', [2, 4, 6, 8, 10, 12], lambda x: dict(L=16, K=x, N=16), 'Number of users K')
R['length'] = sweep('length', [0.5, 1, 2, 4, 8], lambda x: dict(L=16, K=6, N=16, W=float(x)),
                    'Fluid-antenna length W (wavelengths)')
R['aps'] = sweep('aps', [8, 16, 24, 32], lambda x: dict(L=x, K=6, N=16), 'Number of APs L')

# CDF of per-user SE + runtime table
rng = np.random.default_rng(7)
o, tims = run_methods(rng, 16, 6, 16, per_user=True)
fig, ax = plt.subplots(figsize=(3.3, 2.6)); stats = {}
for m, v in o.items():
    v = np.sort(v.ravel()); mk, c = MK[m]
    ax.plot(v, np.arange(1, len(v)+1)/len(v), color=c, label=m, ls=LS[m], marker=mk, ms=3.5, markevery=max(1,len(v)//8), mfc='none')
    stats[m] = dict(mean=float(v.mean()), p5=float(np.percentile(v, 5)), p50=float(np.median(v)), time_ms=tims[m]*1e3)
ax.set_xlabel('Per-user SE (bit/s/Hz)'); ax.set_ylabel('CDF'); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(FIG / 'fig_cdf.png', dpi=600); plt.close(fig)
R['cdf_stats'] = stats

# ablation: candidate list size C in FLARE-LS and top-C accuracy vs greedy label
rng = np.random.default_rng(11); abl = {}
H, Hh, _ = gen_batch(rng, 200 if not args.quick else 60, 16, 6, 16)
y = sel_greedy(Hh); lg = flare.forward(P, flare.features(Hh)); rk = np.argsort(-lg, 2)
g_se = rates(select(H, y), rzf(select(Hh, y))).sum(1).mean()
for C in [1, 2, 3, 5, 8]:
    t = time.time()
    p = rk[:, :, 0] if C == 1 else sel_greedy(Hh, sweeps=1, init=rk[:, :, 0], cand=rk[:, :, :C])
    tt = (time.time()-t)/len(Hh)
    se = rates(select(H, p), rzf(select(Hh, p))).sum(1).mean()
    acc = float((rk[:, :, :C] == y[:, :, None]).any(2).mean())
    abl[C] = dict(topC_acc=acc, se=float(se), ratio=float(se/g_se), time_ms=tt*1e3, evals=0 if C == 1 else 16*C)
    print('abl', C, abl[C])
R['ablation'] = abl
json.dump(R, open(RES / ('main_results_quick.json' if args.quick else 'main_results.json'), 'w'), indent=1)
print(json.dumps(R['cdf_stats'], indent=1))
