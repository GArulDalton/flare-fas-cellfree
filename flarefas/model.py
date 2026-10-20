"""FLARE: permutation-equivariant bipartite GNN for fluid-antenna port selection.

The model is equivariant to permutations of APs, users and ports and therefore
runs on any (L, K, N). It is trained by imitation of Greedy-CD decisions made
on estimated channels. FLARE-LS adds a candidate-restricted local search.
"""
import autograd.numpy as anp
from autograd import grad
import numpy as np
import json, time
from pathlib import Path
from .system import *

D = 32


def init_params(rng, d=D):
    def lin(i, o):
        return [rng.standard_normal((i, o))*np.sqrt(2/i), np.zeros(o)]
    P = {}
    P['A'], P['a'] = lin(3, d)          # per-port encoder
    P['E'], P['e'] = lin(2*d, d)        # edge embedding
    for t in range(2):                  # two message-passing rounds
        P[f'W{t}'], P[f'w{t}'] = lin(5*d, d)
    P['U1'], P['u1'] = lin(3, d)
    P['U2'], _ = lin(d, d)
    P['U3'], _ = lin(2*d, d)
    P['V'], P['v'] = lin(2*d, 1)
    return P


def features(Hhat):
    e = np.log10(np.abs(Hhat)**2 + 1e-3)                      # (B,L,K,N)
    f = np.stack([e/3, e-e.max(3, keepdims=True), e-e.mean(3, keepdims=True)], -1)
    return f.astype(np.float64)


def relu(x):
    return anp.maximum(x, 0)


def forward(P, f):
    """f: (B,L,K,N,3) -> port logits (B,L,N). Permutation-equivariant in APs, users and
    ports; independent of L, K, N."""
    q = relu(anp.einsum('blknc,cd->blknd', f, P['A']) + P['a'])
    h = anp.concatenate([q.mean(3), q.max(3)], -1)             # (B,L,K,2d)
    h = relu(anp.einsum('blkc,cd->blkd', h, P['E']) + P['e'])
    for t in range(2):
        mu = anp.broadcast_to(h.mean(1, keepdims=True), h.shape)   # user context (over APs)
        mx = anp.broadcast_to(h.max(1, keepdims=True), h.shape)
        ma = anp.broadcast_to(h.mean(2, keepdims=True), h.shape)   # AP context (over users)
        mb = anp.broadcast_to(h.max(2, keepdims=True), h.shape)
        z = anp.concatenate([h, mu, mx, ma, mb], -1)
        h = h + relu(anp.einsum('blkc,cd->blkd', z, P[f'W{t}']) + P[f'w{t}'])
    ctx = anp.concatenate([h.mean(1, keepdims=True)*anp.ones_like(h), h], -1)  # user-level context
    s = relu(anp.einsum('blknc,cd->blknd', f, P['U1']) + P['u1']
             + anp.einsum('blkc,cd->blkd', h, P['U2'])[:, :, :, None, :]
             + anp.einsum('blkc,cd->blkd', ctx, P['U3'])[:, :, :, None, :])
    s = anp.concatenate([s.mean(2), s.max(2)], -1)              # aggregate over users -> (B,L,N,2d)
    return (anp.einsum('blnc,co->blno', s, P['V'])[..., 0] + P['v'][0])


def loss_fn(P, f, y):
    lg = forward(P, f)
    lg = lg - lg.max(2, keepdims=True)
    lse = anp.log(anp.exp(lg).sum(2))
    B, L, N = lg.shape
    onehot = np.eye(N)[y]
    return anp.mean(lse - (lg*onehot).sum(2))


def sel_flare(P, Hhat):
    return np.argmax(forward(P, features(Hhat)), 2)


def sel_flare_ls(P, Hhat, C=3):
    lg = forward(P, features(Hhat))
    cand = np.argsort(-lg, 2)[:, :, :C]
    return sel_greedy(Hhat, sweeps=1, init=cand[:, :, 0], cand=cand)


def make_data(rng, n_batches, B=32):
    data = []
    for i in range(n_batches):
        K = rng.choice([4, 6, 8]); L = 16; N = 16
        pdbm = rng.choice([10., 20., 30.])
        H, Hh, _ = gen_batch(rng, B, L, K, N, pilot_dbm=pdbm)
        y = sel_greedy(Hh)
        data.append((features(Hh), y))
    return data


def train(seed=1, n_batches=120, epochs=25, lr=2e-3, out=None, verbose=True):
    rng = np.random.default_rng(seed)
    t = time.time()
    data = make_data(rng, n_batches)
    val = make_data(np.random.default_rng(999), 10)
    if verbose:
        print(f'generated {len(data)} training batches with Greedy-CD labels in {time.time()-t:.1f} s', flush=True)
    P = init_params(rng)
    g = grad(loss_fn)
    m = {k: np.zeros_like(v) for k, v in P.items()}; s2 = {k: np.zeros_like(v) for k, v in P.items()}
    it = 0; hist = []
    for ep in range(epochs):
        order = rng.permutation(len(data)); tl = []
        lr_ep = lr*0.5*(1+np.cos(np.pi*ep/epochs))
        for i in order:
            f, y = data[i]
            G = g(P, f, y); it += 1
            for k in P:
                m[k] = 0.9*m[k]+0.1*G[k]; s2[k] = 0.999*s2[k]+0.001*G[k]**2
                mh = m[k]/(1-0.9**it); sh = s2[k]/(1-0.999**it)
                P[k] = P[k]-lr_ep*mh/(np.sqrt(sh)+1e-8)
            tl.append(loss_fn(P, f, y))
        vl = np.mean([loss_fn(P, f, y) for f, y in val])
        va = np.mean([(np.argmax(forward(P, f), 2) == y).mean() for f, y in val])
        hist.append((ep, float(np.mean(tl)), float(vl), float(va)))
        if verbose:
            print(f'epoch {ep+1:3d}/{epochs}  train CE {hist[-1][1]:.4f}  val CE {hist[-1][2]:.4f}  val top-1 {100*hist[-1][3]:.1f}%  ({time.time()-t:.0f} s)', flush=True)
    if out is not None:
        save_params(P, hist, out)
    return P, hist




def save_params(P, hist, path):
    """Save weights (.npz) and the training history (.json next to it)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in P.items()})
    json.dump([list(h) for h in hist], open(path.with_suffix('.history.json'), 'w'), indent=1)


def load_params(path):
    """Load weights saved by save_params. Returns (params, history)."""
    path = Path(path)
    z = np.load(path)
    P = {k: z[k] for k in z.files}
    hp = path.with_suffix('.history.json')
    hist = [tuple(h) for h in json.load(open(hp))] if hp.exists() else []
    return P, hist


def n_params(P):
    return int(sum(np.asarray(v).size for v in P.values()))
