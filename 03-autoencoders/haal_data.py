"""Haalt de creditcard-fraudedataset één keer op en cachet een compacte versie.

Draai dit één keer na de installatie:  .venv/bin/python haal_data.py
Daarna heeft de notebook geen internet meer nodig.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
DOEL = DATA / "creditcard-subset.parquet"
RANDOM_STATE = 42

def main():
    DATA.mkdir(exist_ok=True)
    if DOEL.exists():
        print(f"Bestaat al: {DOEL}")
        return 0

    from sklearn.datasets import fetch_openml
    print("Dataset ophalen van OpenML (eenmalig, ~150 MB)...")
    ds = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    df = ds.frame
    print(f"  binnengehaald: {df.shape[0]:,} transacties x {df.shape[1]} kolommen")

    kolom = "Class" if "Class" in df.columns else df.columns[-1]
    df[kolom] = df[kolom].astype(int)

    fraude = df[df[kolom] == 1]
    normaal = df[df[kolom] == 0].sample(n=60_000, random_state=RANDOM_STATE)
    subset = (pd.concat([normaal, fraude])
                .sample(frac=1, random_state=RANDOM_STATE)      # door elkaar husselen
                .reset_index(drop=True))

    subset.to_parquet(DOEL, index=False)
    mb = DOEL.stat().st_size / 1e6
    print(f"\nOpgeslagen: {DOEL}  ({mb:.1f} MB)")
    print(f"  {len(subset):,} rijen, waarvan {int(subset[kolom].sum())} fraude "
          f"({subset[kolom].mean():.3%})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
