"""
STEP 4 - LOOK AT THE MISTAKES   (error analysis)

A good student does not only count the mistakes - he LOOKS at them.
This script draws a picture with two rows of photos from the FINAL EXAM (test photos):
   row 1: MISSED potholes  - the model said "fine", but there IS a pothole
                             (sorted: the ones the model was MOST sure about being fine come first)
   row 2: FALSE ALARMS     - the model said "pothole!", but the label says there is none
                             (sorted: the ones the model was MOST sure about come first)
It also prints how many mistakes happen on photos that have cracks.

How to run (Terminal, inside the road-project folder, .venv on):
    python 04_error_analysis.py resnet
    python 04_error_analysis.py small
(the name must be one that exists in the 'results' folder, like  resnet  or  resnet_nopretrain)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = Path("cache")
RESULTS = Path("results")
HOW_MANY = 8          # photos per row


def main():
    if len(sys.argv) < 2:
        print("Tell me which result to study, like this:  python 04_error_analysis.py resnet")
        sys.exit(1)
    name = sys.argv[1]
    pred_file = RESULTS / f"{name}_test_predictions.csv"
    if not pred_file.exists():
        print("I cannot find", pred_file)
        print("Results available:", sorted(p.name.replace("_test_predictions.csv", "")
                                           for p in RESULTS.glob("*_test_predictions.csv")))
        sys.exit(1)

    pred = pd.read_csv(pred_file)
    meta = pd.read_csv(CACHE / "meta.csv")
    row_of = {n: i for i, n in enumerate(meta["name"])}          # photo name -> row in the cache
    pred["row"] = pred["name"].map(row_of)
    pred = pred.merge(meta[["name", "has_crack"]], on="name", how="left")
    images = np.load(CACHE / "images.npy", mmap_mode="r")

    missed = pred[(pred["truth"] == 1) & (pred["model_says"] == 0)].sort_values("chance_of_pothole")
    alarms = pred[(pred["truth"] == 0) & (pred["model_says"] == 1)].sort_values("chance_of_pothole", ascending=False)
    caught = pred[(pred["truth"] == 1) & (pred["model_says"] == 1)]
    fine = pred[(pred["truth"] == 0) & (pred["model_says"] == 0)]

    print(f"Result studied: {name}")
    print(f"   caught potholes : {len(caught)}")
    print(f"   missed potholes : {len(missed)}")
    print(f"   false alarms    : {len(alarms)}")
    print(f"   correct 'fine'  : {len(fine)}")
    print()
    print("Do cracks confuse the model?")
    print(f"   photos WITHOUT pothole that have cracks: false-alarm rate "
          f"{100 * alarms['has_crack'].sum() / max(int(((pred['truth'] == 0) & (pred['has_crack'] == 1)).sum()), 1):.1f} %")
    print(f"   photos WITHOUT pothole and NO cracks   : false-alarm rate "
          f"{100 * (len(alarms) - alarms['has_crack'].sum()) / max(int(((pred['truth'] == 0) & (pred['has_crack'] == 0)).sum()), 1):.1f} %")
    print(f"   of the {len(missed)} missed potholes, {int(missed['has_crack'].sum())} photos also have cracks")

    rows = [("MISSED potholes (model said fine)", missed), ("FALSE ALARMS (model said pothole)", alarms)]
    fig, axes = plt.subplots(2, HOW_MANY, figsize=(2.6 * HOW_MANY, 7.6), squeeze=False)
    for r, (title, frame) in enumerate(rows):
        for c in range(HOW_MANY):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(frame):
                item = frame.iloc[c]
                ax.imshow(images[int(item["row"])])
                ax.set_title(f"chance {item['chance_of_pothole']:.2f}\n{item['name']}", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, hspace=0.55, wspace=0.05)
    for r, (title, frame) in enumerate(rows):                    # a big title above each row
        box = axes[r][0].get_position()
        fig.text(box.x0, box.y1 + 0.07, title, fontsize=12, fontweight="bold")
    out = RESULTS / f"{name}_errors.png"
    plt.savefig(out, dpi=90)
    print("\nSaved the picture:", out, "   (open it and LOOK at the mistakes!)")


if __name__ == "__main__":
    main()
