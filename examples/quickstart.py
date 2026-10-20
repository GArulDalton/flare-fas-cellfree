"""Quick start: compare port-selection schemes on 120 random networks (~10 s)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flarefas as F

P, _ = F.load_params(ROOT / 'weights' / 'flare_params.npz')
rng = np.random.default_rng(2026)
L, K, N = 16, 6, 16
H, Hh, _ = F.gen_batch(rng, 120, L, K, N, W=2.0, pilot_dbm=20.0)

schemes = {
    'Fixed port': F.sel_fixed(Hh),
    'Max-gain': F.sel_maxgain(Hh),
    'Greedy-CD': F.sel_greedy(Hh),
    'FLARE': F.sel_flare(P, Hh),
    'FLARE-LS': F.sel_flare_ls(P, Hh),
}
print(f'L={L} APs, K={K} users, N={N} ports, 120 realizations')
for name, ports in schemes.items():
    se = F.rates(F.select(H, ports), F.rzf(F.select(Hh, ports))).sum(1).mean()
    print(f'{name:12s} average sum SE = {se:6.2f} bit/s/Hz')
