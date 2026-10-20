"""System model for fluid-antenna-enabled cell-free massive MIMO.

Network geometry, 3GPP UMi large-scale fading, Jakes-correlated fluid-antenna
ports, LMMSE port-channel estimation (full and port-subsampled sounding),
RZF precoding with per-AP power constraints, SE evaluation and baseline
port-selection schemes (Fixed, Random, Max-gain, Greedy-CD).
"""
import numpy as np
from scipy.special import j0

# ---------------- system parameters ----------------
FC = 3.5            # GHz
AREA = 400.0        # m (square side)
H_DIFF = 10.0       # AP-UE height difference (m)
P_AP_DBM = 23.0     # per-AP transmit power
NOISE_DBM = -174 + 10*np.log10(20e6) + 7   # 20 MHz, NF 7 dB -> ~-94 dBm
SHADOW = 8.2
P_AP = 10**((P_AP_DBM-30)/10)
SIG2 = 10**((NOISE_DBM-30)/10)


def port_corr(N, W):
    """Jakes spatial correlation between N ports on a linear fluid antenna of length W wavelengths."""
    pos = np.arange(N) * (W/(N-1) if N > 1 else 0)
    D = np.abs(pos[:, None]-pos[None, :])
    return j0(2*np.pi*D)


def sqrtm_psd(R):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 0, None)
    return (V*np.sqrt(w)) @ V.conj().T


def gen_batch(rng, B, L, K, N, W=2.0, pilot_dbm=20.0):
    """Returns true channels H (B,L,K,N) and LMMSE estimates Hhat, normalised by noise std.
    Channel entries are scaled so |h|^2 = beta*P/sigma^2 (i.e. SNR-normalised)."""
    ap = rng.uniform(0, AREA, (B, L, 2))
    ue = rng.uniform(0, AREA, (B, K, 2))
    d = np.sqrt(((ap[:, :, None, :]-ue[:, None, :, :])**2).sum(-1) + H_DIFF**2)
    pl = 32.4 + 20*np.log10(FC) + 31.9*np.log10(d)
    beta_db = -pl + SHADOW*rng.standard_normal((B, L, K))
    beta = 10**(beta_db/10) * P_AP / SIG2          # SNR-normalised large-scale gain
    R = port_corr(N, W)
    Rh = sqrtm_psd(R)
    g = (rng.standard_normal((B, L, K, N)) + 1j*rng.standard_normal((B, L, K, N)))/np.sqrt(2)
    H = np.sqrt(beta)[..., None] * (g @ Rh.T)
    # uplink pilot observation on every port (pilot sweeps across ports):
    # y = sqrt(p_pilot) h_raw + n  ->  in SNR-normalised units  y' = h + sqrt(P_AP/p_pilot) n
    n = (rng.standard_normal(H.shape)+1j*rng.standard_normal(H.shape))/np.sqrt(2)
    p_pilot = 10**((pilot_dbm-30)/10)
    sp = np.sqrt(P_AP/p_pilot)
    Y = H + sp*n
    # LMMSE per link: Hhat = beta R (beta R + sp^2 I)^-1 y
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 0, None)
    bw = beta[..., None]*w                           # (B,L,K,N)
    filt = bw/(bw + sp**2)
    Yv = Y @ V.conj()                                 # project on eigenbasis (V real)
    Hhat = (Yv*filt) @ V.T
    return H, Hhat, beta


def select(Hfull, ports):
    """Hfull (B,L,K,N), ports (B,L) -> effective K x L channel (B,K,L)."""
    B, L, K, N = Hfull.shape
    idx = np.broadcast_to(ports[:, :, None, None], (B, L, K, 1))
    return np.take_along_axis(Hfull, idx, axis=3)[..., 0].transpose(0, 2, 1)


def rzf(Hh):
    """Hh (B,K,L). Regularised ZF with global scaling to meet per-AP power (normalised P=1)."""
    B, K, L = Hh.shape
    A = Hh @ Hh.conj().transpose(0, 2, 1) + K*np.eye(K)[None]
    Wp = Hh.conj().transpose(0, 2, 1) @ np.linalg.inv(A)       # (B,L,K)
    Wp = Wp/np.linalg.norm(Wp, axis=1, keepdims=True)           # unit-norm per user
    rowp = (np.abs(Wp)**2).sum(2)                               # per-AP power
    s = 1/np.sqrt(rowp.max(1))
    return Wp*s[:, None, None]


def rates(H, Wp):
    """True rates per user: H (B,K,L), Wp (B,L,K)."""
    G = np.abs(H @ Wp)**2              # (B,K,K) [k, j]
    sig = np.einsum('bkk->bk', G)
    intf = G.sum(2)-sig
    return np.log2(1+sig/(intf+1))


def sumrate_est(Hh):
    return rates(Hh, rzf(Hh)).sum(1)


# ---------------- port selection baselines ----------------
def sel_fixed(Hhat):
    B, L, K, N = Hhat.shape
    return np.full((B, L), N//2, int)


def sel_random(Hhat, rng):
    B, L, K, N = Hhat.shape
    return rng.integers(0, N, (B, L))


def sel_maxgain(Hhat):
    return np.argmax((np.abs(Hhat)**2).sum(2), axis=2)


def sel_greedy(Hhat, sweeps=2, init=None, cand=None):
    """Centralised coordinate-descent port selection (benchmark / label generator).
    cand: optional (B,L,C) candidate port lists (for FLARE-LS)."""
    B, L, K, N = Hhat.shape
    ports = sel_maxgain(Hhat) if init is None else init.copy()
    for _ in range(sweeps):
        for l in range(L):
            cl = np.arange(N)[None].repeat(B, 0) if cand is None else cand[:, l, :]
            C = cl.shape[1]
            best = np.full(B, -np.inf); bestp = ports[:, l].copy()
            base = select(Hhat, ports)     # (B,K,L)
            for c in range(C):
                Ht = base.copy()
                Ht[:, :, l] = Hhat[np.arange(B), l, :, cl[:, c]]
                r = sumrate_est(Ht)
                m = r > best
                best[m] = r[m]; bestp[m] = cl[m, c]
            ports[:, l] = bestp
    return ports


def gen_batch_sub(rng, B, L, K, N, No, W=2.0, pilot_dbm=20.0):
    """Port-subsampled training: only No evenly spaced ports are sounded; the remaining
    ports are reconstructed by LMMSE interpolation through the Jakes correlation R."""
    ap = rng.uniform(0, AREA, (B, L, 2)); ue = rng.uniform(0, AREA, (B, K, 2))
    d = np.sqrt(((ap[:, :, None, :]-ue[:, None, :, :])**2).sum(-1) + H_DIFF**2)
    pl = 32.4 + 20*np.log10(FC) + 31.9*np.log10(d)
    beta = 10**((-pl + SHADOW*rng.standard_normal((B, L, K)))/10) * P_AP / SIG2
    R = port_corr(N, W); Rh = sqrtm_psd(R)
    g = (rng.standard_normal((B, L, K, N)) + 1j*rng.standard_normal((B, L, K, N)))/np.sqrt(2)
    H = np.sqrt(beta)[..., None] * (g @ Rh.T)
    S = np.unique(np.round(np.linspace(0, N-1, No)).astype(int)) if No > 1 else np.array([N//2])
    sp2 = P_AP/10**((pilot_dbm-30)/10)
    n = (rng.standard_normal(H[..., S].shape)+1j*rng.standard_normal(H[..., S].shape))/np.sqrt(2)
    Y = H[..., S] + np.sqrt(sp2)*n
    RS = R[:, S]; RSS = R[np.ix_(S, S)]
    w, V = np.linalg.eigh(RSS); w = np.clip(w, 0, None)
    bw = beta[..., None]*w
    Yv = (Y @ V) * (beta[..., None]/(bw + sp2))
    Hhat = Yv @ (RS @ V).T
    return H, Hhat, beta
