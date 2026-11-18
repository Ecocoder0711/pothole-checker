"""
STEP 2 - PREPARE THE DATA   (Road-damage project)

What this script does, in simple words:
  1. SHRINK every photo to SIZE x SIZE pixels and keep them all in ONE file (cache/images.npy).
     Why? Reading 7,706 big photos again and again is slow. A shrunk copy in one file is fast.
  2. SPLIT the photos into 3 groups (like school):
        train      (70 %)  = the lesson book. The model studies these.
        validation (15 %)  = practice quiz. We use it to choose settings while studying.
        test       (15 %)  = FINAL EXAM. We look at it only at the very end, once.
  3. Show the LAZY baseline ("always say: no pothole") on the final-exam photos.

Why split "in blocks"?
  Photos taken one after another on the same road look almost the same.
  If one photo is in the lesson book and its twin is in the final exam, the model
  can just remember the twin. That is cheating by accident, and it is called LEAKAGE.
  So we keep photos that are close in number together, in blocks of 50, and put a
  whole block into ONE group.

How to run (Terminal, inside the road-project folder, .venv on):
    python 02_prepare_data.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

SIZE = 320           # photos are shrunk to 320 x 320 (potholes are small, so we do not shrink too much)
SEED = 42            # fixed number -> the same random split every time (reproducible)
BLOCK = 50           # photos that stay together
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15      # the remaining 15 % is the test set
PHOTO_FOLDER = Path("rdd_india")
CACHE = Path("cache")


def photo_number(name):
    """India_000123.jpg  ->  123   (the number inside the file name)"""
    found = re.findall(r"\d+", name)
    return int(found[0]) if found else 0


def main():
    if not Path("labels.csv").exists():
        print("I cannot find labels.csv. Run step 1 first:  python 01_look_at_data.py rdd_india")
        sys.exit(1)
    if not PHOTO_FOLDER.exists():
        print("I cannot find the folder", PHOTO_FOLDER, "- are you inside the road-project folder?")
        sys.exit(1)

    df = pd.read_csv("labels.csv")

    # ---------- 1. find the photos by NAME, so it still works if the folder was moved ----------
    where = {}
    for p in PHOTO_FOLDER.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "__MACOSX" not in p.parts:
            where[p.name] = p
    df["name"] = df["image"].map(lambda p: Path(p).name)
    missing = ~df["name"].isin(where.keys())
    if missing.any():
        print(f"{int(missing.sum())} photos from labels.csv are missing in {PHOTO_FOLDER}. Re-run step 1.")
        sys.exit(1)
    df["path"] = df["name"].map(lambda n: str(where[n]))
    df = df.sort_values("name").reset_index(drop=True)      # fixed order -> reproducible

    # ---------- 2. split in blocks ----------
    df["block"] = df["name"].map(photo_number) // BLOCK
    blocks = np.array(sorted(df["block"].unique()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(blocks)                                       # shuffle the BLOCKS, not single photos
    n_train = int(len(blocks) * TRAIN_FRAC)
    n_val = int(len(blocks) * VAL_FRAC)
    split_of_block = {}
    for i, b in enumerate(blocks):
        if i < n_train:
            split_of_block[b] = "train"
        elif i < n_train + n_val:
            split_of_block[b] = "val"
        else:
            split_of_block[b] = "test"
    df["split"] = df["block"].map(split_of_block)

    # ---------- 3. shrink all photos and keep them in ONE file ----------
    CACHE.mkdir(exist_ok=True)
    print(f"Shrinking {len(df)} photos to {SIZE}x{SIZE} ... (about 2-4 minutes)")
    images = np.lib.format.open_memmap(CACHE / "images.npy", mode="w+",
                                       dtype=np.uint8, shape=(len(df), SIZE, SIZE, 3))
    for i, path in enumerate(df["path"]):
        img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        images[i] = np.asarray(img)
        if (i + 1) % 1000 == 0:
            print(f"   {i + 1} / {len(df)}")
    images.flush()
    del images

    np.save(CACHE / "labels.npy", df["has_pothole"].to_numpy(dtype=np.int64))
    df[["name", "path", "has_pothole", "has_crack", "split", "block"]].to_csv(CACHE / "meta.csv", index=False)
    with open(CACHE / "config.json", "w") as f:
        json.dump({"size": SIZE, "seed": SEED, "block": BLOCK,
                   "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC}, f, indent=2)

    # ---------- 4. show the groups ----------
    print("\nThe three groups:")
    print(f"{'group':<12}{'photos':>8}{'with pothole':>14}{'  % pothole':>12}")
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name]
        pot = int(part["has_pothole"].sum())
        print(f"{name:<12}{len(part):>8}{pot:>14}{100 * pot / max(len(part), 1):>11.1f}%")

    # ---------- 5. the LAZY baseline on the final-exam (test) photos ----------
    test = df[df["split"] == "test"]
    y_true = test["has_pothole"].to_numpy()
    y_pred = np.zeros_like(y_true)                            # lazy: ALWAYS says "no pothole"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("\nLAZY BASELINE ('always say no pothole') on the TEST photos:")
    print(f"   accuracy  : {100 * accuracy_score(y_true, y_pred):.1f} %   <- looks good, but...")
    print(f"   precision : {precision_score(y_true, y_pred, zero_division=0):.3f}   (it never says pothole, so nothing to be right about)")
    print(f"   recall    : {recall_score(y_true, y_pred, zero_division=0):.3f}   <- it catches NO potholes")
    print(f"   F1        : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\n   Confusion matrix (rows = truth, columns = what the model said):")
    print("                         says NO    says POTHOLE")
    print(f"   really NO pothole   {tn:>8}    {fp:>8}")
    print(f"   really a POTHOLE    {fn:>8}    {tp:>8}")
    print("\nDone! Data is ready in the folder:", CACHE)


if __name__ == "__main__":
    main()
