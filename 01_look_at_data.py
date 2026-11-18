"""
STEP 1 - LOOK AT THE DATA  (Road-damage project)

What this script does, in simple words:
  1. Finds all the road photos in the folder you give it.
  2. Reads the little "label" file (.xml) that comes with each photo.
     The label file says what damage is in the photo and where (boxes).
       D00 = long crack    D10 = sideways crack
       D20 = "alligator" crack (cracks that look like scales)
       D40 = POTHOLE
  3. Makes ONE simple question for each photo:  "does it contain a pothole?"  (yes/no)
  4. Saves the answers in a table file called labels.csv
  5. Prints how many photos are yes / no, and saves a picture (samples.png)
     so you can look at real photos with your own eyes.

How to run (in Terminal, inside the road-project folder, with .venv on):
    python 01_look_at_data.py  <path to the unzipped India folder>
"""

import sys
import xml.etree.ElementTree as ET   # tool for reading .xml label files
from collections import Counter      # tool for counting things
from pathlib import Path             # tool for working with folders/files

import pandas as pd                  # tool for tables (like Excel inside Python)
import matplotlib
matplotlib.use("Agg")                # lets us SAVE pictures without opening a window
import matplotlib.pyplot as plt
from PIL import Image                # tool for opening photos

# Different copies of this dataset spell the label names differently,
# so we accept the official codes AND plain words (small letters, to be safe).
POTHOLE_NAMES = {"d40", "pothole", "potholes"}
CRACK_NAMES = {"d00", "d10", "d20", "crack", "cracks", "longitudinal crack",
               "transverse crack", "alligator crack", "aligator crack"}
SEED = 42                            # fixed number so random choices are the same every run


def main():
    # ---------- 0. check what the user typed ----------
    if len(sys.argv) < 2:
        print("Please tell me the folder, like this:")
        print("   python 01_look_at_data.py ~/road-project/India")
        sys.exit(1)
    root = Path(sys.argv[1]).expanduser()
    if not root.exists():
        print("I cannot find this folder:", root)
        sys.exit(1)

    # ---------- 1. find all photos ----------
    # We store them as  {file name without .jpg : full path}
    photos = {}
    for p in root.rglob("*"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue                 # skip junk files that macOS sometimes makes
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            photos[p.stem] = p
    print("Photos found (all folders):", len(photos))

    # ---------- 2. read every label file ----------
    rows = []                        # one row per photo
    label_counter = Counter()        # counts "in how many photos does D40 appear?"
    broken = 0
    for xml_path in sorted(root.rglob("*.xml")):
        if "__MACOSX" in xml_path.parts or xml_path.name.startswith("._"):
            continue
        if xml_path.stem not in photos:
            continue                 # a label with no photo -> ignore
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            broken += 1              # a damaged label file -> skip it
            continue
        names = [o.findtext("name", default="").strip()
                 for o in tree.getroot().iter("object")]
        for n in set(names):
            label_counter[n] += 1
        small = [n.lower() for n in names]                       # small letters, to compare safely

        rows.append({
            "image": str(photos[xml_path.stem].resolve()),   # full path, so later steps never lose it
            "has_pothole": int(any(n in POTHOLE_NAMES for n in small)),   # 1 = yes, 0 = no
            "has_crack": int(any(n in CRACK_NAMES for n in small)),
            "n_objects": len(names),
            "classes": " ".join(sorted(set(names))),
        })

    if not rows:
        print("I found no photo+label pairs. Is this the right folder? (the one with 'train' inside)")
        sys.exit(1)

    # ---------- 3. make the table and save it ----------
    df = pd.DataFrame(rows)
    df.to_csv("labels.csv", index=False)

    # ---------- 4. print the summary ----------
    total = len(df)
    pot = int(df["has_pothole"].sum())
    crack_only = int(((df["has_crack"] == 1) & (df["has_pothole"] == 0)).sum())
    nothing = int((df["n_objects"] == 0).sum())
    print()
    print("Photos WITH a label file :", total, f"(broken label files skipped: {broken})")
    print(f"Photos with a POTHOLE    : {pot}  ({100 * pot / total:.1f} %)")
    print(f"Photos with NO pothole   : {total - pot}  ({100 * (total - pot) / total:.1f} %)")
    print("   of those, cracks only :", crack_only)
    print("   of those, no damage at all:", nothing)
    print()
    print("Every label name seen (number of photos that contain it):")
    for name, count in label_counter.most_common():
        print("   ", name, count)
    print("\nSaved the table to labels.csv")

    # ---------- 5. save a picture of sample photos ----------
    yes = df[df["has_pothole"] == 1]
    no = df[df["has_pothole"] == 0]
    k = min(6, len(yes), len(no))
    if k > 0:
        yes = yes.sample(k, random_state=SEED)
        no = no.sample(k, random_state=SEED)
        fig, axes = plt.subplots(2, k, figsize=(3 * k, 6.4), squeeze=False)
        for col in range(k):
            for row, (title, frame) in enumerate([("POTHOLE", yes), ("no pothole", no)]):
                ax = axes[row][col]
                ax.imshow(Image.open(frame.iloc[col]["image"]).convert("RGB"))
                ax.set_title(title, fontsize=11)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig("samples.png", dpi=90)
        print("Saved a picture of sample photos to samples.png  (open it and LOOK!)")


if __name__ == "__main__":
    main()
