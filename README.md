# Pothole checker: "does this road photo have a pothole?"

Individual Deep Learning PBL project (Python, PyTorch). Binary image classification on Indian road photos.

## 1. Problem

Given one road photo, answer **pothole / no pothole**.
This is a scaled-down version of "Urban Pothole and Road Damage Assessment System" (project 23 in the catalogue):
the full system would also locate each pothole with a box. Here we only classify the whole photo, so the project can be
studied and explained completely by one student.

## 2. Data

* **RDD2022 (India part)**, from the CRDDC 2022 release (figshare). 7,706 photos that come with label files.
* Label files (Pascal VOC XML) mark damage types. Code `D40` = pothole; `D00`, `D10`, `D20` = cracks.
* A photo counts as **positive** if it has at least one `D40` box. About 20 % of the photos are positive.
* Photos are shrunk to 320 x 320.
* Licence: the images are CC BY-SA 4.0 (check the dataset page before any commercial use).

**Split (blocks, to avoid leakage).** Photos with close numbers come from the same road and look alike.
So we keep photos in blocks of 50 consecutive numbers and put each whole block into one group:

| group | photos | purpose |
|---|---|---|
| train (70 %) | 5,367 | the model learns from these |
| validation (15 %) | 1,131 | choose the best epoch and the alarm level |
| test (15 %) | 1,208 | final exam, used once per experiment, never used for choosing anything |

## 3. Method

* **Baseline:** "always say no pothole" (the lazy model).
* **Model A, small CNN from scratch:** 4 blocks of Conv, BatchNorm, ReLU and MaxPool, then Linear. 98,049 parameters.
* **Model B, ResNet18 with ready-made ImageNet weights, fine-tuned (transfer learning):** last layer replaced by one output. 11,177,025 parameters.
* Loss: `BCEWithLogitsLoss` with `pos_weight` about 4 (a missed pothole counts 4x). Optimizer: Adam
  (learning rate 0.001 small / 0.0001 ResNet), batch 32, 8 epochs (ResNet) / 12 epochs (small).
* Augmentation while studying: random left-right flip and +-20 % brightness.
* The best epoch is chosen by validation Average Precision. The **alarm level** (decision threshold) is chosen on the
  validation photos by maximum F1. The test photos are never used for choosing.
* Seed 42 everywhere that matters.

## 4. Results (test photos, 1,208 photos, 241 of them with a pothole)

| run | what is different | F1 | precision | recall | accuracy |
|---|---|---|---|---|---|
| lazy baseline | always "no pothole" | 0.000 | – | 0.000 | 80.0 % |
| small | CNN from scratch | 0.520 | 0.635 | 0.440 | 83.8 % |
| resnet (**final model**) | ResNet18, ready-made weights | **0.690** | 0.723 | 0.660 | 88.2 % |
| resnet_noweight | no class weight | 0.690 | 0.696 | 0.685 | 87.8 % |
| resnet_noaug | no augmentation | 0.676 | 0.793 | 0.589 | 88.7 % |
| resnet_lr0.001 | learning rate 0.001 | 0.599 | 0.609 | 0.589 | 84.3 % |
| resnet_nopretrain | no ready-made weights | 0.563 | 0.598 | 0.531 | 83.5 % |

Final model confusion matrix: 159 potholes found, 82 missed, 61 false alarms, 906 correct "no pothole".

Rough 95 % ranges of F1 (from re-drawing the exam photos 1,000 times) are about +-0.05, so runs closer than that are
tied. See `results/comparison.csv` and `results/comparison.png`.

**What the experiments show**

* Ready-made weights matter a lot (0.563 to 0.690). A ResNet taught from zero is hardly better than the small CNN.
* Small learning steps matter (0.0001 beats 0.001).
* Class weight and augmentation made **no measurable difference** here (all three ResNet variants are tied).
* The final model was fixed (`resnet`) before the experiments and was not swapped afterwards.

**Error analysis** (`python 04_error_analysis.py resnet`, picture in `results/resnet_errors.png`)

* On photos without a pothole, the false-alarm rate is 9.5 % when the photo has cracks and 5.1 % when it has none:
  cracks confuse the model.
* 82 potholes were missed; 35 of those photos also contain cracks.

## 5. Limitations (honest)

* The model misses about 1 in 3 potholes (recall 0.66). It is a student project, not a safety tool.
* One dataset, one split, one run per setting. Small differences between runs are not meaningful.
* Labels are not perfect, and small potholes shrink a lot at 320 x 320.
* Only whole-photo yes/no. No location of the pothole, no size or severity.
* Training on the Apple GPU (MPS) is not perfectly repeatable, so a re-run may differ by a little.

## 6. How to reproduce

Python 3.9, Apple silicon Mac (also works on CPU, only slower).

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python 00_extract_india.py ~/Downloads/RDD2022_released_through_CRDDC2022.zip   # take the India photos out of the big zip
python 01_look_at_data.py rdd_india                                              # labels.csv + samples.png
python 02_prepare_data.py                                                        # shrink photos, split in blocks, lazy baseline
python 03_train.py small                                                         # small CNN from scratch
python 03_train.py resnet                                                        # about 15 min on an M-series Mac (needs internet the first time for the ResNet18 weights)
python 03_train.py resnet --no-pretrained                                        # experiments (each saved under its own name)
python 03_train.py resnet --no-weight
python 03_train.py resnet --no-augment
python 03_train.py resnet --lr 0.001
python 04_error_analysis.py resnet                                               # picture of the mistakes
python 05_compare_results.py                                                     # scoreboard table + chart
```

Using the trained model (no internet needed):

```
python 06_predict.py path/to/photo_or_folder
pip install streamlit
streamlit run 07_app.py --server.fileWatcherType none --browser.gatherUsageStats false
```

## 7. Files

| file | job |
|---|---|
| `00_extract_india.py` | take the India photos out of the big zip |
| `01_look_at_data.py` | read the labels, make `labels.csv`, show sample photos |
| `02_prepare_data.py` | shrink photos, block split, lazy baseline |
| `03_train.py` | the two models, training, best epoch, alarm level, final exam |
| `04_error_analysis.py` | picture of missed potholes and false alarms |
| `05_compare_results.py` | scoreboard table and chart |
| `pothole_model.py` | helper: load the saved brain, answer for one photo |
| `06_predict.py` | command-line predictor |
| `07_app.py` | small web page (GUI) |
| `results/resnet.pt` | the trained final model |

## 8. AI assistance disclosure

An AI assistant (Claude, by Anthropic) was used as a study and coding partner in this project. It helped with
choosing the project scope, writing the first versions of the scripts, and explaining concepts. I ran all the
steps and experiments on my own Mac, made the decisions together with the assistant.

## 9. Credits

* RDD2022 dataset: Arya, Maeda, Ghosh, Toshniwal, Sekimoto, "RDD2022: A multi-national image dataset for automatic
  Road Damage Detection" (arXiv:2209.08538), and the CRDDC 2022 release on figshare.
* ResNet: He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", CVPR 2016. Weights: torchvision (ImageNet).
* PyTorch, torchvision, scikit-learn, pandas, matplotlib, Streamlit.
