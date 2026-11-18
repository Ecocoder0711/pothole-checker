"""
STEP 7 - THE LITTLE WEB PAGE (GUI)   "Pothole checker"

A bonus for the demo. It runs on YOUR Mac only (no internet needed) and opens in the browser.
Two tabs:
   1. "Check my own photo"       - upload a road photo, the brain answers
   2. "Try a final-exam photo"   - pick a random photo the brain never studied; the page shows the
                                    real answer next to the brain's answer, so everybody can see right/wrong

One-time setup (Terminal, inside road-project, .venv on):
    pip install streamlit
How to run:
    streamlit run 07_app.py --server.fileWatcherType none --browser.gatherUsageStats false
(Stop it with Ctrl+C in the Terminal.)
"""

import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import pothole_model

st.set_page_config(page_title="Pothole checker")
st.title("Pothole checker")
st.write("Show me a road photo. My brain (a ResNet18 taught on Indian road photos) "
         "tells you if it thinks there is a pothole.")


@st.cache_resource                       # load the brain only once, not on every click
def get_brain():
    return pothole_model.load("resnet")


@st.cache_resource
def get_images():
    return np.load("cache/images.npy", mmap_mode="r")      # all shrunk photos from step 2


@st.cache_data
def get_meta():
    return pd.read_csv("cache/meta.csv")                   # name, label and group of every photo


brain = get_brain()


def show_verdict(chance, says_pothole):
    st.metric("Chance of pothole", f"{chance:.0%}")
    if says_pothole:
        st.error("POTHOLE spotted")
    else:
        st.success("No pothole")
    st.caption(f"The brain shouts 'pothole' when the chance is at least {brain['threshold']:.0%} "
               "(this alarm level was chosen on practice-quiz photos).")


tab_upload, tab_exam = st.tabs(["Check my own photo", "Try a final-exam photo"])

# ---------------------------------------------------------------- tab 1: your own photo
with tab_upload:
    file = st.file_uploader("Choose a road photo (jpg or png)", type=["jpg", "jpeg", "png"])
    if file is not None:
        photo = Image.open(file)
        chance, says = pothole_model.ask(brain, photo)
        preview = ImageOps.exif_transpose(photo).convert("RGB")
        preview.thumbnail((900, 900))                      # keep the page light for big phone photos
        st.image(preview, caption="your photo")
        show_verdict(chance, says)

# ---------------------------------------------------------------- tab 2: a final-exam photo
with tab_exam:
    st.write("These photos were kept hidden while the brain studied, so this is a fair test.")
    meta = get_meta()
    images = get_images()
    exam_rows = meta.index[meta["split"] == "test"].to_numpy()

    kind = st.radio("Which photos?", ["any", "only real potholes", "only photos with no pothole"],
                    horizontal=True)
    if kind == "only real potholes":
        rows = exam_rows[meta.loc[exam_rows, "has_pothole"].to_numpy() == 1]
    elif kind == "only photos with no pothole":
        rows = exam_rows[meta.loc[exam_rows, "has_pothole"].to_numpy() == 0]
    else:
        rows = exam_rows

    clicked = st.button("Show a random photo")
    if clicked or st.session_state.get("exam_pick") not in set(rows.tolist()):
        st.session_state["exam_pick"] = int(random.choice(rows.tolist()))

    i = st.session_state["exam_pick"]
    array = np.array(images[i])                            # a copy of this one photo
    truth = bool(meta.loc[i, "has_pothole"])
    chance, says = pothole_model.ask(brain, Image.fromarray(array))
    st.image(array, caption=f"{meta.loc[i, 'name']}   |   real answer (from the label file): "
                            f"{'POTHOLE' if truth else 'no pothole'}")
    show_verdict(chance, says)
    if says == truth:
        st.info("The brain is RIGHT on this photo.")
    else:
        st.warning("The brain is WRONG on this photo. (It does happen - see the error analysis.)")

# ---------------------------------------------------------------- honest numbers
with st.expander("How good is this brain? (final-exam numbers)"):
    info = brain["info"]
    total = info["tp"] + info["fn"] + info["fp"] + info["tn"]
    st.write(f"On {total} exam photos: F1 {info['f1']:.2f}, precision {info['precision']:.2f}, "
             f"recall {info['recall']:.2f}, accuracy {100 * info['accuracy']:.1f} %.")
    st.write(f"It found {info['tp']} of {info['tp'] + info['fn']} real potholes and raised "
             f"{info['fp']} false alarms.")
    st.write("A lazy model that always says 'no pothole' gets F1 0.00. "
             "This is a student project - do not use it for real road-safety decisions.")
