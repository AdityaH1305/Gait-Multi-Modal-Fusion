# Getting Started

Welcome to the setup guide for the Modified GaitSet-Based Multimodal Gait Recognition Framework.

This guide provides detailed instructions for configuring the environment, preparing the CASIA-B dataset, training the model, and evaluating its performance.

---

## Contents

1. Installation
2. System Requirements
3. Dataset Preparation
4. Usage

---

## Installation

Follow the steps below to set up the project on your local machine.

### 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/AdityaH1305/Gait-Multi-Modal-Fusion.git
cd Gait-Multi-Modal-Fusion
```

### 2. Create a Virtual Environment

It is recommended to use a dedicated Python virtual environment.

```bash
python -m venv gait_env
```

Activate the environment.

**Windows**

```bash
gait_env\Scripts\activate
```

**Linux / macOS**

```bash
source gait_env/bin/activate
```

### 3. Install Dependencies

Install all required Python packages.

```bash
pip install -r requirements.txt
```

### 4. Verify the Installation

Confirm that Python and PyTorch are installed correctly.

```bash
python --version
python -c "import torch; print(torch.__version__)"
```

---

## System Requirements

The project was developed and tested using the following software and hardware configuration.

### Software Requirements

| Component | Version |
|-----------|---------|
| Python | 3.11.15 |
| PyTorch | 2.7.1 |
| CUDA Toolkit | 11.8 |
| TorchVision | Compatible with PyTorch 2.7.1 |
| NumPy | Latest stable version |
| OpenCV | Latest stable version |
| Pillow | Latest stable version |
| Matplotlib | Latest stable version |

### Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Memory | 6 GB VRAM |
| CUDA Version | 11.8 |

### Recommended Environment

For the best compatibility and reproducibility, the following setup is recommended:

- Python 3.11
- CUDA 11.8
- NVIDIA GPU with CUDA support
- Windows 10/11 or a recent Linux distribution

The project can run on CPU, but GPU acceleration is strongly recommended given the computational demands of training a deep learning model.

---

## Dataset Preparation

### CASIA-B Dataset

This project uses the CASIA-B Gait Dataset, one of the most widely used benchmark datasets for gait recognition research.

> **Note:** The CASIA-B dataset is not included in this repository due to licensing restrictions. It must be obtained separately from the official source.

### Directory Structure

After downloading the dataset, organize it as follows:

```text
Gait-Multi-Modal-Fusion/
│
├── GaitDatasetB-silh/
│   ├── 001/
│   ├── 002/
│   ├── ...
│
├── preprocess.py
├── pack_npy.py
└── ...
```

### Step 1 — Preprocess the Dataset

Run the preprocessing script:

```bash
python preprocess.py
```

CASIA-B ships with silhouettes already extracted, so this step normalizes them rather than segmenting them. For each frame the pipeline:

- Crops to the bounding box of the non-zero foreground pixels
- Resizes to 64 × 64 using cubic interpolation
- Averages all frames in a sequence into a Gait Energy Image (GEI)

The normalized silhouettes and the GEI are written per sequence.

> **Known limitation.** Cropping to the bounding box and resizing to a square does not preserve aspect ratio. Because the bounding box widens and narrows across the gait cycle as the limbs swing, each frame is stretched by a different factor, and subject height is discarded. Height-preserving, centroid-aligned normalization is planned work.

### Step 2 — Package the Dataset

Once preprocessing is complete, package the processed data into NumPy arrays.

```bash
python pack_npy.py
```

This script converts the processed dataset into an optimized format that can be efficiently loaded during model training and evaluation.

### Expected Dataset Layout

After preprocessing and packing, each sequence lives in its own `subject / condition / angle` folder:

```text
Processed_CASIAB/
├── 001/                        # subject (001-124)
│   ├── nm-01/                  # condition: nm-01..06, bg-01..02, cl-01..02
│   │   ├── 000/                # camera angle (000, 018, ... 180)
│   │   │   ├── frame_001.png   # normalized silhouettes, 64x64
│   │   │   ├── ...
│   │   │   ├── 001_nm_000_GEI.png
│   │   │   ├── frames.npy      # (N, 64, 64) uint8  <- written by pack_npy.py
│   │   │   └── gei.npy         # (64, 64)   uint8  <- written by pack_npy.py
│   │   └── ...
│   └── ...
└── ...
```

Training and evaluation read the `.npy` files exclusively, since loading a single array is far faster than reading roughly 80 individual PNGs per sequence.

Subjects 001–074 form the training split; 075–124 are held out for testing.

---

## Usage

The complete workflow for training and evaluating the proposed multimodal gait recognition framework is outlined below.

### Step 1 — Preprocess the Dataset

Extract silhouettes, perform spatial normalization, and generate Gait Energy Images (GEIs).

```bash
python preprocess.py
```

### Step 2 — Package the Dataset

Convert the processed dataset into NumPy arrays for efficient loading during training.

```bash
python pack_npy.py
```

### Step 3 — Train the Model

Train the modified GaitSet model with multimodal feature fusion.

```bash
python train.py
```

During training, the framework:

- Extracts frame-set features using the dynamic branch, with Set Pooling across frames
- Extracts GEI features using the static branch
- Applies spatial attention for multimodal feature fusion
- Optimizes cross-entropy on the classifier head together with a batch-hard triplet loss on the 256-D embedding, using PK-structured batches of P identities × K sequences
- Saves the final checkpoint to `results/fused_gait_model.pth`

> **Note.** There is currently no validation split, so the checkpoint written is the one from the last epoch rather than the best-scoring one.

### Step 4 — Evaluate the Model

Evaluate the trained model on the held-out test subjects (075–124).

```bash
python eval.py
```

`eval.py` is the only script that loads the model. It runs a single GPU pass and writes `results/embeddings.npz`, which the two reporting scripts then consume, so the model never runs more than once per evaluation.

The evaluation reports:

- Rank-1 accuracy per walking condition (NM, BG, CL)
- The full 11 × 11 cross-view matrix, with the identical-view diagonal excluded
- A same-view reference figure for comparison
- A sanity check on genuine versus impostor similarity

Useful flags:

```bash
python eval.py --show-matrix           # print the full 11x11 matrices
python eval.py --frame-budget 1536     # raise if you have more than 6 GB VRAM
python eval.py --weights path/to.pth   # evaluate a different checkpoint
```

> **Note on VRAM.** Batching is driven by a total frame budget rather than a fixed batch size, because the dynamic branch reshapes to `(B x N, 1, 64, 64)` before the first convolution — memory therefore scales with sequences × frames, not sequences. The default of 768 peaks near 3.7 GB on a 6 GB card.

### Step 5 — Plot the Cross-View Matrix

```bash
python plot_view_matrix.py
```

Renders `results/cross_view_accuracy_matrix.png`, one 11 × 11 heatmap per walking condition on a shared color scale.

### Step 6 — Compute Biometric Metrics

Generate verification metrics from the same embeddings.

```bash
python compute_biometrics.py
```

This script computes:

- ROC curves per condition plus a combined curve
- Area Under the Curve (AUC)
- Equal Error Rate (EER) and the threshold at which it occurs
- Genuine versus impostor score distributions

Identical-view pairs are excluded by default to match the Rank-1 protocol; pass `--include-same-view` to include them.

### Typical Workflow

The complete execution pipeline is shown below.

```text
Download CASIA-B
        │
        ▼
Preprocess Dataset ......... preprocess.py
        │
        ▼
Package Dataset ............ pack_npy.py
        │
        ▼
Train Model ................ train.py    -> results/fused_gait_model.pth
        │
        ▼
Evaluate Model ............. eval.py     -> results/embeddings.npz
        │                                   (single GPU pass)
        ├──────────────────────────┐
        ▼                          ▼
Cross-view heatmap          ROC / AUC / EER
plot_view_matrix.py         compute_biometrics.py
```

### Output Files

| Path | Produced by | Contents |
|:-----|:------------|:---------|
| `results/fused_gait_model.pth` | `train.py` | Trained model weights |
| `results/training_curves.png` | `train.py` | Cross-entropy loss, triplet loss, accuracy |
| `results/attention_maps/` | `train.py` | Attention masks captured every 10 epochs |
| `results/embeddings.npz` | `eval.py` | Gallery, probe and template embeddings, labels, cross-view matrices |
| `results/cross_view_accuracy_matrix.png` | `plot_view_matrix.py` | 11 × 11 heatmaps per condition |
| `results/gait_verification_roc.png` | `compute_biometrics.py` | ROC curves |
| `results/score_distributions.png` | `compute_biometrics.py` | Genuine versus impostor score histograms |

`results/embeddings.npz` is a derived artifact and is git-ignored; regenerate it by re-running `eval.py`.
