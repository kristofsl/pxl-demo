"""Haalt MNIST op en cachet een compacte selectie.

Draai dit één keer na de installatie:  .venv/bin/python haal_data.py
Daarna heeft de notebook geen internet meer nodig.
"""
import sys
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data"
DOEL = DATA / "cijfers.npz"
AANTAL = 20_000
RANDOM_STATE = 42

def main():
    DATA.mkdir(exist_ok=True)
    if DOEL.exists():
        print(f"Bestaat al: {DOEL}")
        return 0

    from sklearn.datasets import fetch_openml
    print("MNIST ophalen van OpenML (eenmalig)...")
    ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = ds.data.astype("float32"), ds.target.astype(int)
    print(f"  binnengehaald: {X.shape[0]:,} cijfers")

    rng = np.random.default_rng(RANDOM_STATE)
    keuze = rng.choice(len(X), AANTAL, replace=False)
    np.savez_compressed(DOEL, X=X[keuze].astype("uint8"), y=y[keuze].astype("int8"))

    mb = DOEL.stat().st_size / 1e6
    print(f"\nOpgeslagen: {DOEL}  ({mb:.1f} MB)")
    print(f"  {AANTAL:,} cijfers van 28x28")
    return 0

if __name__ == "__main__":
    sys.exit(main())
