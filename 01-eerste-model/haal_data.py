"""Haalt de Adult/Census-dataset op en cachet hem lokaal.

Draai dit één keer na de installatie:  .venv/bin/python haal_data.py
Daarna heeft de notebook geen internet meer nodig.
"""
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
DOEL = DATA / "inkomen.parquet"

def main():
    DATA.mkdir(exist_ok=True)
    if DOEL.exists():
        print(f"Bestaat al: {DOEL}")
        return 0

    from sklearn.datasets import fetch_openml
    print("Dataset ophalen van OpenML (eenmalig)...")
    ds = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = ds.frame
    print(f"  binnengehaald: {df.shape[0]:,} rijen x {df.shape[1]} kolommen")

    df.to_parquet(DOEL, index=False)
    print(f"\nOpgeslagen: {DOEL}  ({DOEL.stat().st_size / 1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
