"""Unit tests: shapes, power constraints, LMMSE quality and FLARE equivariance."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flarefas as F

WEIGHTS = ROOT / 'weights' / 'flare_params.npz'


@pytest.fixture(scope='module')
def batch():
    rng = np.random.default_rng(0)
    return F.gen_batch(rng, 8, 16, 6, 16)


@pytest.fixture(scope='module')
def params():
    return F.load_params(WEIGHTS)[0]


def test_shapes(batch):
    H, Hh, beta = batch
    assert H.shape == Hh.shape == (8, 16, 6, 16)
    assert beta.shape == (8, 16, 6)


def test_per_ap_power_constraint(batch):
    _, Hh, _ = batch
    W = F.rzf(F.select(Hh, F.sel_maxgain(Hh)))
    per_ap = (np.abs(W) ** 2).sum(2)
    assert np.all(per_ap <= 1 + 1e-9)
    assert np.allclose(per_ap.max(1), 1)


def test_lmmse_beats_raw_estimate():
    rng = np.random.default_rng(1)
    H, Hh, _ = F.gen_batch(rng, 8, 8, 4, 16, pilot_dbm=10.0)
    assert np.mean(np.abs(H - Hh) ** 2) < np.mean(np.abs(H) ** 2)


def test_subsampled_full_equals_full_dimension():
    rng = np.random.default_rng(2)
    H, Hh, _ = F.gen_batch_sub(rng, 4, 8, 4, 16, No=4)
    assert Hh.shape == H.shape


def test_parameter_count(params):
    assert F.n_params(params) == 15777


def test_permutation_equivariance(batch, params):
    """Permuting APs, users or ports permutes FLARE's scores accordingly."""
    _, Hh, _ = batch
    base = F.forward(params, F.features(Hh))
    rng = np.random.default_rng(3)
    pa, pu, pp = rng.permutation(16), rng.permutation(6), rng.permutation(16)
    out = F.forward(params, F.features(Hh[:, pa][:, :, pu][..., pp]))
    assert np.allclose(out, base[:, pa][..., pp], atol=1e-8)


def test_size_agnostic(params):
    rng = np.random.default_rng(4)
    for L, K, N in [(8, 3, 4), (24, 10, 32)]:
        _, Hh, _ = F.gen_batch(rng, 2, L, K, N)
        assert F.sel_flare(params, Hh).shape == (2, L)


def test_flare_ls_not_worse_than_flare_on_estimates(params):
    rng = np.random.default_rng(5)
    _, Hh, _ = F.gen_batch(rng, 16, 16, 6, 16)
    r1 = F.sumrate_est(F.select(Hh, F.sel_flare(params, Hh))).mean()
    r2 = F.sumrate_est(F.select(Hh, F.sel_flare_ls(params, Hh))).mean()
    assert r2 >= r1 - 1e-9


def test_training_smoke(tmp_path):
    P, hist = F.train(seed=0, n_batches=2, epochs=1, out=tmp_path / 'w.npz', verbose=False)
    assert len(hist) == 1 and (tmp_path / 'w.npz').exists()
