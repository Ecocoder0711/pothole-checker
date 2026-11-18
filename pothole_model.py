"""
HELPER - load the trained brain and ask it about ONE photo.

Used by  06_predict.py  (Terminal)  and  07_app.py  (the little web page).
It works with NO internet: the brain's numbers are read from the file  results/resnet.pt.

It re-uses the SAME brain classes and the SAME photo-preparation as training (from 03_train.py),
so a new photo is treated exactly like the exam photos were.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

train = importlib.import_module("03_train")      # (a name that starts with a number needs import_module)

RESULTS = Path("results")
CACHE = Path("cache")


def pick_device():
    """the Mac's GPU ("mps") if there is one, otherwise the normal processor"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(name="resnet"):
    """open the saved brain called `name` (a file in the results folder) and get it ready to answer"""
    weights_file = RESULTS / f"{name}.pt"
    info_file = RESULTS / f"{name}_results.json"
    if not weights_file.exists() or not info_file.exists():
        raise SystemExit(f"I cannot find {weights_file} or {info_file}. "
                         "Are you inside the road-project folder? Did the training finish?")
    with open(info_file) as f:
        info = json.load(f)                       # holds the alarm level and the final-exam scores

    device = pick_device()
    if name.startswith("small"):
        model = train.SmallCNN()
    else:
        model = train.ResNetBinary(pretrained=False)      # empty shell; our own saved numbers go in next
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)
    model.eval()                                  # "exam mode": no flipping, no learning

    size = 320                                    # the size used in step 2
    config = CACHE / "config.json"
    if config.exists():
        with open(config) as f:
            size = json.load(f)["size"]
    return {"model": model, "device": device, "threshold": info["threshold"],
            "size": size, "name": name, "info": info}


def ask(brain, photo):
    """photo = a PIL image.  Returns (chance of pothole from 0 to 1,  True if the brain shouts 'pothole')"""
    photo = ImageOps.exif_transpose(photo).convert("RGB")       # phone photos: turn them the right way up
    photo = photo.resize((brain["size"], brain["size"]), Image.Resampling.BILINEAR)   # same shrinking as step 2
    batch = torch.from_numpy(np.array(photo)).unsqueeze(0)      # 1 photo: (1, height, width, 3)
    with torch.no_grad():                                       # no learning now, only answering
        logit = brain["model"](train.prepare(batch, brain["device"]))
    chance = torch.sigmoid(logit)[0].item()                     # turn the brain's raw number into 0..1
    return chance, bool(chance >= brain["threshold"])
