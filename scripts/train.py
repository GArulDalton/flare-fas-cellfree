"""Train FLARE by imitation of Greedy-CD (about 12 minutes on one CPU core)."""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flarefas import model as flare

ap = argparse.ArgumentParser(description='Train FLARE port selector.')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--batches', type=int, default=120, help='training batches of 32 realizations')
ap.add_argument('--epochs', type=int, default=25)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--out', default=str(ROOT / 'weights' / 'flare_params.npz'))
args = ap.parse_args()
P, hist = flare.train(seed=args.seed, n_batches=args.batches, epochs=args.epochs, lr=args.lr, out=args.out)
print(f'saved {flare.n_params(P)} parameters to {args.out}')
