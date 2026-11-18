"""
STEP 6 - ASK THE TRAINED BRAIN ABOUT NEW PHOTOS   (Terminal, works with no internet)

Give it photo files and/or folders. For each photo it prints the chance of a pothole and its verdict.

How to run (Terminal, inside the road-project folder, .venv on):
    python 06_predict.py  ~/Desktop/some_road_photo.jpg
    python 06_predict.py  photo1.jpg photo2.jpg
    python 06_predict.py  ~/Desktop/my_photos_folder
    python 06_predict.py  photo.jpg --model small        # use the small brain instead of ResNet
"""

import argparse
from pathlib import Path

from PIL import Image

import pothole_model

PHOTO_ENDINGS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description="Does this road photo have a pothole?")
    ap.add_argument("photos", nargs="+", help="photo files and/or folders")
    ap.add_argument("--model", default="resnet", help="which saved brain to use (default: resnet)")
    args = ap.parse_args()

    brain = pothole_model.load(args.model)
    print(f"Brain: {args.model} | alarm level: {brain['threshold']:.0%} "
          f"(it shouts 'pothole' when its chance is at least this)\n")

    files = []
    for item in args.photos:
        path = Path(item).expanduser()
        if path.is_dir():
            files += sorted(p for p in path.rglob("*")
                            if p.suffix.lower() in PHOTO_ENDINGS and not p.name.startswith("._"))
        else:
            files.append(path)
    if not files:
        print("I found no photos. Check the path.")
        return

    shouted = 0
    for path in files:
        try:
            photo = Image.open(path)
            photo.load()                                  # really read it now, so a broken file fails here
        except Exception as err:
            print(f"{path.name:<34} cannot open this file ({err})")
            continue
        chance, says = pothole_model.ask(brain, photo)
        shouted += int(says)
        print(f"{path.name:<34} chance of pothole {chance:6.1%}   ->  {'POTHOLE' if says else 'no pothole'}")
    print(f"\n{shouted} of {len(files)} photos got the answer POTHOLE.")


if __name__ == "__main__":
    main()
