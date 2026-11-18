"""
STEP 3 - TEACH THE MODELS   (Road-damage project: "does this photo have a pothole?")

Two brains to compare (plus the lazy baseline from step 2):
   small   : a small brain that WE build ourselves from scratch (4 layers of "looking")
   resnet  : ResNet18, a ready-made brain that already learned to see on 1 million photos.
             We only teach it our new job. This is called TRANSFER LEARNING (or fine-tuning).

How to run (Terminal, inside the road-project folder, .venv on):
    python 03_train.py small  --quick       # tiny fast test (2 epochs, few photos) - run this FIRST
    python 03_train.py resnet --quick
    python 03_train.py small                # the real training
    python 03_train.py resnet               # the real training
    python 03_train.py resnet --no-pretrained   # experiment: same ResNet18 but starting EMPTY-headed
    python 03_train.py resnet --no-weight       # experiment: no extra weight on missed potholes
    python 03_train.py resnet --no-augment      # experiment: no flipping / brightness changes
    python 03_train.py resnet --lr 0.001        # experiment: bigger dial turns (default 0.0001)
    (every experiment is saved under its own name in the 'results' folder, nothing gets overwritten)

What happens, in simple words:
   1. Load the shrunk photos (from step 2) and the 3 groups: train / validation / test.
   2. For each epoch (one full pass over all the lesson-book photos):
        a. look at a small batch of photos and GUESS "pothole or not" (forward pass)
        b. measure how WRONG the guesses were (the LOSS)
        c. work out which way to turn every dial to be less wrong (backprop = gradients)
        d. turn every dial a little bit (the OPTIMIZER step)
      then take a practice quiz on the validation photos.
   3. Keep the weights of the best epoch (best validation score).
   4. Pick the "alarm threshold" using the validation photos (NOT the test photos).
   5. Open the final exam (test photos) ONCE and write down precision, recall, F1.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # if the Mac GPU lacks a tool, use the CPU for it

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score)

CACHE = Path("cache")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # the numbers ResNet was trained with
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------------------
#  THE BRAINS
# ----------------------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Our own small brain. Each block = look for patterns (Conv), tidy numbers (BatchNorm),
    keep only strong signals (ReLU), shrink the picture to half (MaxPool)."""

    def __init__(self):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)       # average everything into one number per pattern
        self.head = nn.Linear(128, 1)             # final decision: ONE number (high = pothole)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(1)


class ResNetBinary(nn.Module):
    """ResNet18 with its last layer swapped for a new one that gives ONE number."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision import models
        if pretrained:
            try:
                net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # downloads ~45 MB the first time
            except Exception as err:                                              # no internet etc.
                raise SystemExit("Could not download the ready-made ResNet18 weights (is the internet on?).\n"
                                 f"Error: {err}")
        else:
            net = models.resnet18(weights=None)
        net.fc = nn.Linear(512, 1)
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(1)


# ----------------------------------------------------------------------------------------
#  SMALL HELPERS
# ----------------------------------------------------------------------------------------
def prepare(batch_uint8, device, augment=False):
    """photos (0..255, height-width-colour)  ->  numbers the brain likes (colour-height-width, normalised).
    augment=True makes small random changes (flip, brightness) so the model cannot just memorise."""
    x = batch_uint8.to(device).permute(0, 3, 1, 2).contiguous().float() / 255.0
    if augment:
        flip = torch.rand(x.shape[0], device=device) < 0.5                         # half the photos get mirrored
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        bright = 1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1, device=device) * 2 - 1)   # +-20 % brightness
        x = (x * bright).clamp(0, 1)
    return (x - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def predict_probs(model, images, device, batch_size=64):
    """chance (0..1) that each photo has a pothole"""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        logits = model(prepare(images[i:i + batch_size], device))
        out.append(torch.sigmoid(logits).cpu())
    return torch.cat(out).numpy()


def best_threshold(y_true, probs):
    """the alarm level that gives the best F1 on these photos"""
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    k = int(np.argmax(f1))
    return float(thr[k]), float(f1[k])


def score(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_true, probs)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ----------------------------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["small", "resnet"])
    ap.add_argument("--epochs", type=int, default=None, help="how many passes over the lesson book")
    ap.add_argument("--lr", type=float, default=None, help="learning rate = how big each dial turn is")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true", help="resnet only: start from empty instead of ready-made")
    ap.add_argument("--no-weight", action="store_true", help="experiment: treat a missed pothole like a false alarm (no extra weight)")
    ap.add_argument("--no-augment", action="store_true", help="experiment: no flipping / brightness changes while studying")
    ap.add_argument("--quick", action="store_true", help="tiny fast run to test that everything works")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ----- settings -----
    epochs = args.epochs or (2 if args.quick else (12 if args.model == "small" else 8))
    lr = args.lr or (1e-3 if args.model == "small" else 1e-4)
    # every experiment gets its OWN name, so results never overwrite each other
    name = args.model
    if args.no_pretrained:
        name += "_nopretrain"
    if args.no_weight:
        name += "_noweight"
    if args.no_augment:
        name += "_noaug"
    if args.lr:
        name += f"_lr{args.lr:g}"
    if args.epochs:
        name += f"_ep{args.epochs}"
    if args.quick:
        name += "_quick"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {name} | device: {device} | epochs: {epochs} | learning rate: {lr} | batch: {args.batch_size}")

    # ----- load the data prepared in step 2 -----
    if not (CACHE / "images.npy").exists():
        raise SystemExit("I cannot find cache/images.npy. Run step 2 first:  python 02_prepare_data.py")
    meta = pd.read_csv(CACHE / "meta.csv")
    labels = np.load(CACHE / "labels.npy")
    all_images = np.load(CACHE / "images.npy", mmap_mode="r")

    def get_split(split, quick_limit):
        idx = np.where(meta["split"].to_numpy() == split)[0]
        if args.quick and len(idx) > quick_limit:
            idx = np.sort(rng.choice(idx, quick_limit, replace=False))
        return idx, torch.from_numpy(np.array(all_images[idx], dtype=np.uint8)), labels[idx]

    tr_idx, X_tr, y_tr = get_split("train", 640)
    va_idx, X_va, y_va = get_split("val", 256)
    te_idx, X_te, y_te = get_split("test", 256)
    print(f"photos -> train: {len(y_tr)} | validation: {len(y_va)} | test: {len(y_te)}")

    # ----- the brain, the loss, the optimizer -----
    model = SmallCNN() if args.model == "small" else ResNetBinary(pretrained=not args.no_pretrained)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"The brain has {n_params:,} dials (parameters)")

    # Only ~20 % of photos have potholes. pos_weight makes a MISSED pothole count more
    # than a false alarm while studying, so the model does not become lazy.
    n_pos, n_neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
    weight_value = 1.0 if args.no_weight else n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([weight_value], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)      # "how wrong" for a yes/no question
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    # decides how to turn the dials
    print(f"pos_weight = {pos_weight.item():.2f}  (a missed pothole counts {pos_weight.item():.1f}x more while studying)")
    print(f"settings -> ready-made weights: {args.model == 'resnet' and not args.no_pretrained} | "
          f"class weight: {not args.no_weight} | augmentation: {not args.no_augment}")

    y_tr_t = torch.from_numpy(y_tr).float()
    history, best_ap, best_epoch = [], -1.0, 0
    weights_path = out_dir / f"{name}.pt"

    # ----- the learning loop -----
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(y_tr))                      # new random order every epoch
        total_loss, seen = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            if len(b) < 2:                                     # BatchNorm needs at least 2 photos
                continue
            xb = prepare(X_tr[b], device, augment=not args.no_augment)
            yb = y_tr_t[b].to(device)

            logits = model(xb)                                 # a. guess
            loss = loss_fn(logits, yb)                         # b. how wrong?
            optimizer.zero_grad()
            loss.backward()                                    # c. which way to turn each dial (gradients)
            optimizer.step()                                   # d. turn the dials a little

            total_loss += loss.item() * len(b)
            seen += len(b)

        train_loss = total_loss / seen
        val_probs = predict_probs(model, X_va, device)         # practice quiz
        val_ap = float(average_precision_score(y_va, val_probs))
        _, val_f1 = best_threshold(y_va, val_probs)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_ap": round(val_ap, 4), "val_best_f1": round(val_f1, 4)})
        mark = ""
        if val_ap > best_ap:                                   # best practice-quiz score so far -> save it
            best_ap, best_epoch = val_ap, epoch
            torch.save(model.state_dict(), weights_path)
            mark = "  <- best so far, saved"
        print(f"epoch {epoch:>2}/{epochs} | train loss {train_loss:.4f} | val AP {val_ap:.3f} | "
              f"val best F1 {val_f1:.3f} | {time.time() - t0:.0f}s{mark}")

    # ----- final exam, ONCE, with the best weights -----
    model.load_state_dict(torch.load(weights_path, map_location=device))
    val_probs = predict_probs(model, X_va, device)
    threshold, _ = best_threshold(y_va, val_probs)             # threshold chosen on VALIDATION, never on test
    test_probs = predict_probs(model, X_te, device)
    res = score(y_te, test_probs, threshold)
    res.update({"model": name, "best_epoch": best_epoch, "epochs": epochs, "learning_rate": lr,
                "batch_size": args.batch_size, "seed": args.seed, "parameters": n_params,
                "pretrained": (args.model == "resnet" and not args.no_pretrained),
                "class_weight": not args.no_weight, "augmentation": not args.no_augment,
                "device": str(device), "train_photos": int(len(y_tr)), "test_photos": int(len(y_te)),
                "history": history})
    with open(out_dir / f"{name}_results.json", "w") as f:
        json.dump(res, f, indent=2)

    pd.DataFrame({"name": meta["name"].to_numpy()[te_idx], "truth": y_te,
                  "chance_of_pothole": np.round(test_probs, 4),
                  "model_says": (test_probs >= threshold).astype(int)}
                 ).to_csv(out_dir / f"{name}_test_predictions.csv", index=False)

    # ----- one picture with everything -----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ep = [h["epoch"] for h in history]
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (how wrong)")
    if len(ep) <= 20:
        ax[0].set_xticks(ep)                                   # whole numbers only
    ax2 = ax[0].twinx()
    ax2.plot(ep, [h["val_ap"] for h in history], marker="s", color="tab:orange", label="validation AP")
    ax2.set_ylabel("validation AP (higher = better)")
    ax[0].set_title("Learning curve")
    prec, rec, _ = precision_recall_curve(y_te, test_probs)
    ax[1].plot(rec, prec)
    ax[1].scatter([res["recall"]], [res["precision"]], color="red", zorder=3, label="our chosen alarm level")
    ax[1].axhline(float(y_te.mean()), color="gray", linestyle="--", label="lazy guess level")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision vs recall (test)")
    ax[1].legend(loc="upper right", fontsize=8)
    cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
    ax[2].imshow(cm, cmap="Blues")
    for r in range(2):
        for c in range(2):
            ax[2].text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=16,
                       color="white" if cm[r, c] > cm.max() / 2 else "black")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["says NO", "says POTHOLE"])
    ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["really NO", "really POTHOLE"])
    ax[2].set_title("Confusion matrix (test)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_report.png", dpi=110)

    # ----- print the final exam result -----
    print("\n" + "=" * 62)
    print(f"FINAL EXAM (test photos) for {name}   [best epoch: {best_epoch}, alarm level: {res['threshold']}]")
    print(f"   accuracy  : {100 * res['accuracy']:.1f} %")
    print(f"   precision : {res['precision']:.3f}   (when it says pothole, how often is it right)")
    print(f"   recall    : {res['recall']:.3f}   (of all real potholes, how many it found)")
    print(f"   F1        : {res['f1']:.3f}   (lazy baseline: 0.000)")
    print(f"   confusion : caught {res['tp']} | missed {res['fn']} | false alarms {res['fp']} | correct 'fine' {res['tn']}")
    print("=" * 62)
    print(f"Saved in the folder 'results': {name}.pt (weights), {name}_results.json, "
          f"{name}_test_predictions.csv, {name}_report.png")
    if args.quick:
        print("\n(This was only the quick test. The numbers mean little. If it ran without errors, you are ready for the real run.)")


if __name__ == "__main__":
    main()
