# 🚀 Getting Started

Welcome to the setup guide for the **Modified GaitSet-Based Multimodal Gait Recognition Framework**.

This guide provides detailed instructions for configuring the environment, preparing the CASIA-B dataset, training the model, and evaluating its performance.

---

## Contents

1. Installation
2. System Requirements
3. Dataset Preparation
4. Usage

---

# ⚙️ Installation

Follow the steps below to set up the project on your local machine.

## 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/AdityaH1305/Gait-Multi-Modal-Fusion.git
cd Gait-Multi-Modal-Fusion
```

---

## 2. Create a Virtual Environment

It is recommended to use a dedicated Python virtual environment.

```bash
python -m venv gait_env
```

Activate the environment.

### Windows

```bash
gait_env\Scripts\activate
```

### Linux / macOS

```bash
source gait_env/bin/activate
```

---

## 3. Install Dependencies

Install all required Python packages.

```bash
pip install -r requirements.txt
```

---

## 4. Verify the Installation

Confirm that Python and PyTorch are installed correctly.

```bash
python --version
python -c "import torch; print(torch.__version__)"

```

---

# 💻 System Requirements

The project was developed and tested using the following software and hardware configuration.

## Software Requirements

| Component | Version |
|-----------|---------|
| Python | 3.11.15 |
| PyTorch | 2.7.1 |
| CUDA Toolkit | 11.8 |
| TorchVision | Compatible with PyTorch 2.7.1 |
| NumPy | Latest Stable Version |
| OpenCV | Latest Stable Version |
| Pillow | Latest Stable Version |
| Matplotlib | Latest Stable Version |

---

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Memory | 6 GB VRAM |
| CUDA Version | 11.8 |

---

## Recommended Environment

For the best compatibility and reproducibility, it is recommended to use:

- Python 3.11
- CUDA 11.8
- NVIDIA GPU with CUDA support
- Windows 10/11 or a recent Linux distribution

Although the project can run on a CPU, GPU acceleration is strongly recommended due to the computational requirements of deep learning model training.

---

# 📂 Dataset Preparation

## CASIA-B Dataset

This project uses the **CASIA-B Gait Dataset**, one of the most widely used benchmark datasets for gait recognition research.

> **Note:** The CASIA-B dataset is **not included** in this repository due to licensing restrictions. It must be obtained separately from the official source.

---

## Directory Structure

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

---

## Step 1 — Preprocess the Dataset

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

---

## Step 2 — Package the Dataset

Once preprocessing is complete, package the processed data into NumPy arrays.

```bash
python pack_npy.py
```

This script converts the processed dataset into an optimized format that can be efficiently loaded during model training and evaluation.

---

## Expected Dataset Layout

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