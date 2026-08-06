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

The preprocessing pipeline performs the following operations automatically:

- Adaptive background subtraction
- Binary silhouette extraction
- Subject localization and cropping
- Centroid alignment
- Spatial normalization (64 × 64 pixels)
- Gait cycle organization
- Gait Energy Image (GEI) generation

The processed silhouettes and GEIs are stored for training.

### Step 2 — Package the Dataset

Once preprocessing is complete, package the processed data into NumPy arrays.

```bash
python pack_npy.py
```

This script converts the processed dataset into an optimized format that can be efficiently loaded during model training and evaluation.

### Expected Dataset Layout

After preprocessing, the project directory should resemble the following structure:

```text
Processed_CASIAB/
├── silhouettes/
├── gei/
├── train/
├── test/
└── *.npy
```

These files are then used by the training and evaluation scripts.

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

- Extracts silhouette features using the spatial branch
- Extracts GEI features using the temporal branch
- Applies channel attention for multimodal feature fusion
- Learns discriminative gait embeddings using Batch-All Triplet Loss
- Saves the best-performing model checkpoints

### Step 4 — Evaluate the Model

Evaluate the trained model on the test dataset.

```bash
python eval.py
```

The evaluation reports:

- Rank-1 recognition accuracy
- Cross-view recognition performance
- NM (normal walking) accuracy
- BG (walking with bag) accuracy
- CL (walking with coat) accuracy

### Step 5 — Compute Biometric Metrics

Generate verification metrics for the trained model.

```bash
python compute_biometrics.py
```

This script computes:

- Receiver Operating Characteristic (ROC) curve
- Area Under the Curve (AUC)
- Equal Error Rate (EER)
- Optimal decision threshold

### Typical Workflow

The complete execution pipeline is shown below.

```text
Download CASIA-B
        │
        ▼
Preprocess Dataset
(preprocess.py)
        │
        ▼
Package Dataset
(pack_npy.py)
        │
        ▼
Train Model
(train.py)
        │
        ▼
Evaluate Model
(eval.py)
        │
        ▼
Compute ROC & EER
(compute_biometrics.py)
```

### Output Files

After training and evaluation, the project generates:

- Trained model checkpoints (`.pth`)
- Training accuracy and loss curves
- Rank-1 recognition results
- ROC curve
- Equal Error Rate (EER)
- Attention map visualizations
- Evaluation logs
