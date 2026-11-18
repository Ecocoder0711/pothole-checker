"""
STEP 5 - THE SCOREBOARD   (compare all the experiments)

What this script does, in simple words:
  1. Opens the result file of every experiment in the 'results' folder.
  2. Puts them in ONE table (best F1 on top) and prints it.
  3. Adds a "rough range" for each F1 number. Why?
        The final exam has only 241 potholes. If the exam had been a slightly different
        set of photos, every score would move a little. The range shows HOW MUCH it could move.
        Two runs whose ranges overlap a lot are basically TIED. Do not say one is better.
  4. Saves the table (results/comparison.csv) and a picture (results/comparison.png).

How the range is made (the "bootstrap" idea, kid version):
  Take the final-exam answer sheet, draw 1,208 photos out of it WITH putting each one back
  (so some photos appear twice, some never), and compute F1 on that "new exam".
  Do this 1,000 times. The middle 95% of those F1 numbers is the rough range.
  (It is a rough tool - photos from the same road are alike, so the true wobble is
   probably a bit bigger than shown.)

How to run (Terminal, inside the road-project folder, .venv on):
    python 05_compare_results.py
    open results/comparison.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
REPEATS = 1000       # how many "new exams" we draw
SEED = 42            # fixed number -> same answer every time (reproducible)

BLUE = "#2F6FB5"     # colour of the bars
GREY = "#9AA0A6"     # colour of the lazy baseline bar
INK = "#222222"      # colour of the text


def recipe(r):
    """a short sentence: what was special about this run?"""
    name = r["model"]
    if name.startswith("small"):
        return "small brain, built from zero"
    text = "ResNet"
    if r.get("pretrained", "nopretrain" not in name):
        text += ", ready-made weights"
    else:
        text += ", NO ready-made weights"
    if not r.get("class_weight", True):
        text += ", no class weight"
    if not r.get("augmentation", True):
        text += ", no augmentation"
    if abs(r.get("learning_rate", 1e-4) - 1e-4) > 1e-12:
        text += f", learning rate {r['learning_rate']:g}"
    if name == "resnet":
        text += "  (our main recipe)"
    return text


def f1_range(truth, says):
    """rough range of F1: 'draw a new exam with put-back' many times"""
    rng = np.random.RandomState(SEED)
    n = len(truth)
    scores = []
    for _ in range(REPEATS):
        pick = rng.randint(0, n, n)                  # n photos, drawn with put-back
        t, s = truth[pick], says[pick]
        tp = np.sum((t == 1) & (s == 1))
        fp = np.sum((t == 0) & (s == 1))
        fn = np.sum((t == 1) & (s == 0))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    files = sorted(RESULTS.glob("*_results.json"))
    if not files:
        print("I found no result files in the 'results' folder. Run 03_train.py first.")
        return

    rows = []
    truth_example = None
    for path in files:
        with open(path) as f:
            r = json.load(f)
        name = r["model"]
        if "quick" in name or r.get("test_photos", 10 ** 9) < 500:
            continue                                  # the tiny test runs do not count
        pred_file = RESULTS / f"{name}_test_predictions.csv"
        if pred_file.exists():
            pred = pd.read_csv(pred_file)
            lo, hi = f1_range(pred["truth"].to_numpy(), pred["model_says"].to_numpy())
            truth_example = pred["truth"].to_numpy()
        else:
            lo = hi = r["f1"]
        rows.append({"run": name, "what_was_different": recipe(r),
                     "f1": r["f1"], "f1_low": round(lo, 3), "f1_high": round(hi, 3),
                     "precision": r["precision"], "recall": r["recall"], "accuracy": r["accuracy"],
                     "caught": r["tp"], "missed": r["fn"], "false_alarms": r["fp"],
                     "best_epoch": r["best_epoch"]})

    if not rows:
        print("Only quick test runs were found. Do the real runs first.")
        return

    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_f1 = table["f1"].max()
    # "tied" = the best run's score is still inside this run's rough range
    table["vs_best"] = ["best" if f == best_f1 else ("tied" if best_f1 <= hi else "worse")
                        for f, hi in zip(table["f1"], table["f1_high"])]
    table.to_csv(RESULTS / "comparison.csv", index=False)

    # the lazy baseline: always says "no pothole"
    lazy = None
    if truth_example is not None:
        lazy = {"run": "lazy", "what_was_different": "always says: no pothole",
                "f1": 0.0, "f1_low": 0.0, "f1_high": 0.0, "precision": float("nan"), "recall": 0.0,
                "accuracy": round(float(np.mean(truth_example == 0)), 4)}

    # ---------- print the table ----------
    print("\nSCOREBOARD  (final-exam photos, sorted by F1)\n")
    print(f"{'run':<22}{'F1':>6}{'  rough range':>16}{'prec':>7}{'recall':>8}{'acc':>7}"
          f"{'caught':>8}{'missed':>8}{'false':>7}   vs best")
    for _, r in table.iterrows():
        print(f"{r['run']:<22}{r['f1']:>6.3f}   {r['f1_low']:.2f} - {r['f1_high']:.2f}  "
              f"{r['precision']:>6.3f}{r['recall']:>8.3f}{100 * r['accuracy']:>6.1f}%"
              f"{int(r['caught']):>8}{int(r['missed']):>8}{int(r['false_alarms']):>7}   {r['vs_best']}")
    if lazy:
        print(f"{'lazy baseline':<22}{0:>6.3f}   {'':>12}{'-':>7}{0:>8.3f}{100 * lazy['accuracy']:>6.1f}%")
    print("\nWhat was different in each run:")
    for _, r in table.iterrows():
        print(f"   {r['run']:<22}{r['what_was_different']}")
    print("\nHow to read 'vs best':  'tied' means the scores are too close to call.")
    print("Do NOT say one tied run is better than another - the exam is too small to know.")

    # ---------- the picture ----------
    plot_rows = table.to_dict("records")
    if lazy:
        plot_rows.append(lazy)
    n = len(plot_rows)
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * n + 1.8))
    y = np.arange(n)[::-1]                            # best run on top
    f1 = np.array([r["f1"] for r in plot_rows])
    lo = np.array([r["f1_low"] for r in plot_rows])
    hi = np.array([r["f1_high"] for r in plot_rows])
    colours = [GREY if r["run"] == "lazy" else BLUE for r in plot_rows]
    ax.barh(y, f1, height=0.5, color=colours)
    ax.errorbar(f1, y, xerr=[f1 - lo, hi - f1], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for yi, f, h in zip(y, f1, hi):
        ax.text(max(h, f) + 0.015, yi, f"{f:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['run']}\n{r['what_was_different']}" for r in plot_rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score on the final-exam photos  (higher is better; the thin line is the rough range)", color=INK)
    ax.set_title("Which experiment did best?", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.xaxis.grid(True, color="#E3E5E8", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out = RESULTS / "comparison.png"
    fig.savefig(out, dpi=110)
    print("\nSaved:", RESULTS / "comparison.csv", "and", out)


if __name__ == "__main__":
    main()
