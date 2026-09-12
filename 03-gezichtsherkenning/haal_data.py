"""Haalt LFW-gezichten op en zet ze als losse JPEG's in data/gezichten/.

Draai dit één keer na de installatie:  .venv/bin/python haal_data.py
Daarna heeft de notebook geen internet meer nodig.

Resultaat:
    data/gezichten/<Naam Persoon>/0001.jpg, 0002.jpg, ...
"""
import sys, shutil
from pathlib import Path
import numpy as np

MAP = Path(__file__).parent / "data" / "gezichten"
AANTAL_PERSONEN = 16          # hoeveel bekende gezichten
MAX_PER_PERSOON = 40          # meer dan dit voegt weinig toe en kost tijd
RANDOM_STATE = 42

def main():
    if MAP.exists() and any(MAP.iterdir()):
        print(f"Bestaat al: {MAP}")
        return 0

    from sklearn.datasets import fetch_lfw_people
    from PIL import Image

    print("LFW ophalen (eenmalig, ~200 MB)...")
    lfw = fetch_lfw_people(min_faces_per_person=25, color=True,
                           slice_=None, resize=0.5, download_if_missing=True)
    print(f"  {len(lfw.images)} foto's van {len(lfw.target_names)} personen")

    # De personen met de meeste foto's, maar niet de allerbekendste alleen:
    # we pakken de top zodat iedereen genoeg voorbeelden heeft.
    tellingen = np.bincount(lfw.target)
    gekozen = np.argsort(tellingen)[::-1][:AANTAL_PERSONEN]

    rng = np.random.default_rng(RANDOM_STATE)
    MAP.mkdir(parents=True, exist_ok=True)
    totaal = 0
    for idx in gekozen:
        naam = str(lfw.target_names[idx])
        rijen = np.where(lfw.target == idx)[0]
        if len(rijen) > MAX_PER_PERSOON:
            rijen = rng.choice(rijen, MAX_PER_PERSOON, replace=False)
        doel = MAP / naam.replace(" ", "_")
        doel.mkdir(exist_ok=True)
        for n, r in enumerate(sorted(rijen), start=1):
            arr = lfw.images[r]
            arr = (arr * 255).astype("uint8") if arr.max() <= 1.01 else arr.astype("uint8")
            Image.fromarray(arr).save(doel / f"{n:04d}.jpg", quality=92)
        totaal += len(rijen)
        print(f"  {naam:28s} {len(rijen):3d} foto's")

    mb = sum(f.stat().st_size for f in MAP.rglob('*.jpg')) / 1e6
    print(f"\nOpgeslagen: {totaal} foto's van {AANTAL_PERSONEN} personen  ({mb:.1f} MB)")
    print(f"Map: {MAP}")

    eigen = MAP.parent / "mijn-gezichten"
    eigen.mkdir(exist_ok=True)
    (eigen / "LEESMIJ.txt").write_text(
        "Zet hier een map met je eigen naam, en daarin een paar foto's van jezelf.\n"
        "Bijvoorbeeld:  mijn-gezichten/Kristof/foto1.jpg\n"
        "De notebook pikt ze automatisch op.\n", encoding="utf-8")
    print(f"Eigen foto's kun je kwijt in: {eigen}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
