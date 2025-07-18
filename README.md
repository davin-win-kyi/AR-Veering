# AR Veering

**Real-time augmented reality crosswalk assistance for Blind & Low-Vision (BLV) pedestrians.**

---

## 1. System Background

Many blind and low-vision (BLV) individuals report that walking outdoors—especially locating, aligning with, and traversing crosswalks—can be risky and stressful. *Crosswalk veering* (drifting out of the intended crossing path), uncertainty about the current walk signal state, moving vehicles or other dynamic obstacles, locating the curb ramp on the far side, and finding the pedestrian push-button can all increase cognitive load and anxiety.

**AR Veering** is a research prototype that delivers proactive, real-time AR feedback about the current crosswalk scenario. The system fuses depth, video, and model‑based perception to surface:

- **Veering relative to crosswalk alignment.**
- **Walk / Don't Walk signal state.**
- **Dynamic objects** (e.g., vehicles, other pedestrians) in or near the crossing path.
- **Pedestrian curbs & curb ramps.**
- **Pedestrian traffic signal buttons.**

By updating users continuously as conditions change, AR Veering aims to improve confidence, situational awareness, and safety during street crossings.

---

## 2. Prerequisites

- Conda (Miniconda or Anaconda)
- Python 3.10 (environment will be created)
- Git
- pip
- HoloLens 2 + [HL2SS](https://github.com/jdibenes/hl2ss) streaming setup
- Google Gemini API credentials 

---

## 3. Quick Start (TL;DR)

```bash
# create & activate env
conda create -n arveering python=3.10
conda activate arveering

# install top-level deps
pip install -r requirements.txt

# clone dependencies
git clone https://github.com/jdibenes/hl2ss.git
git clone https://github.com/google-gemini/cookbook.git

# install Depth-Anything-V2 under hl2ss/viewer and copy AR Veering scripts + weights
cd hl2ss/viewer
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth-anything-v2
cd depth-anything-v2
# copy best.pt + AR Veering scripts listed below into this folder
# then install model reqs
pip install -r requirements.txt
cd ../../..

# install Depth-Anything-V2 under google-gemini/cookbook/quickstarts and copy Gemini demo scripts
cd cookbook/quickstarts
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth-anything-v2
cd depth-anything-v2
# copy best.pt + Get_started_LiveAPI.py + needed hl2ss_* scripts
cd ../../..
```

See the detailed steps below for the exact file copy list.

---

## 4. Detailed Setup Steps

**All commands assume you're in the root of the AR Veering repo unless otherwise noted.**

### 4.1 Create & Activate Conda Environment

```bash
conda create --name arveering python=3.10
conda activate arveering
```

### 4.2 Install Top-Level Python Requirements

Make sure `requirements.txt` exists in the repo root.

```bash
pip install -r requirements.txt
```

### 4.3 Clone Required External Repos

```bash
git clone https://github.com/jdibenes/hl2ss.git
git clone https://github.com/google-gemini/cookbook.git
```

---

### 4.4 Depth-Anything-V2 for HL2SS Pipeline

We vendor a copy of **Depth-Anything-V2** under `hl2ss/viewer/` for the HoloLens streaming + AR Veering runtime pipeline.

```bash
cd hl2ss/viewer
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth-anything-v2
cd depth-anything-v2
```

#### 4.4.1 Add Model Weights

Download `best.pt` from the shared drive and place it here:

- Google Drive folder: [https://drive.google.com/drive/folders/10NA\_Fr7Wg8B29U-RdhrpcmMq7tNaR2hm?usp=sharing](https://drive.google.com/drive/folders/10NA_Fr7Wg8B29U-RdhrpcmMq7tNaR2hm?usp=sharing)
- After download: move or copy into `hl2ss/viewer/depth-anything-v2/best.pt`.

#### 4.4.2 Add AR Veering Scripts

Copy the following project files into this folder (overwriting if they already exist):

```
AR_Veering_stream.py
hl2ss_3dcv.py
hl2ss_dp.py
hl2ss_ds.py
hl2ss_imshow.py
hl2ss_io.py          
hl2ss_lnm.py
hl2ss_mp.py
hl2ss_mt.py
hl2ss_mx.py
hl2ss_rus.py
hl2ss_sa.py
hl2ss_utilities.py
hl2ss.py
```

#### 4.4.3 Install Depth-Anything-V2 Requirements

From within `hl2ss/viewer/depth-anything-v2`:

```bash
pip install -r requirements.txt
```

Return to repo root:

```bash
cd ../../..
```

---

### 4.5 Depth-Anything-V2 for Gemini Quickstarts

A second copy of **Depth-Anything-V2** is used inside the Google Gemini cookbook quickstart examples

```bash
cd cookbook/quickstarts
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth-anything-v2
cd depth-anything-v2
```

#### 4.5.1 Add Model Weights

Copy the same `best.pt` weights file into this folder.

#### 4.5.2 Add Gemini Demo + HL2SS Scripts

Copy:

```
Get_started_LiveAPI.py
hl2ss_3dcv.py
hl2ss_imshow.py
hl2ss_lnm.py
hl2ss_mp.py
hl2ss_rus.py
hl2ss.py
```

(Include any additional helper modules you rely on.)

Return to repo root when done:

```bash
cd ../../..
```

---



## 6. Running AR Veering (HL2SS Pipeline)

From `hl2ss/viewer/depth-anything-v2` (or wherever you placed `AR_Veering_stream.py`):

```bash
python AR_Veering_stream.py 
```

---

## 7. Running the Gemini Live Quickstart

From `cookbook/quickstarts/depth-anything-v2`:

```bash
python Get_started_LiveAPI.py --mode screen
```

Before running, ensure you have valid Google Gemini credentials. Replace <API_KEY> in AR_Veering_stream.py and Get_started_LiveAPI.py with 
valid google api credentials

---

## 8. Acknowledgements

- **HL2SS** streaming interface for HoloLens 2.
- **Depth-Anything-V2** depth estimation model.
- **Google Gemini Cookbook** sample integrations with multimodal LLM APIs.
