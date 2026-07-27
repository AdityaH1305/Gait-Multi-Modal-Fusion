# 🚶 Advanced Cross-View Gait Recognition using Multimodal Fusion Networks

### A Deep Learning Framework for Robust Human Identification using a Modified GaitSet Architecture and Multimodal Feature Fusion

---

## 📖 Overview

Human gait is a unique behavioural biometric that enables the identification of individuals based solely on their walking pattern. Unlike traditional biometric modalities such as facial recognition or fingerprint analysis, gait recognition can identify subjects from a distance without requiring active user cooperation, making it highly suitable for intelligent surveillance, forensic investigations, and security applications.

This repository presents a complete implementation of a **Modified GaitSet-based Multimodal Gait Recognition Framework**, inspired by the research paper **"Research on Gait Recognition Based on GaitSet and Multimodal Fusion."**

The proposed system extends the original **GaitSet** architecture by integrating **Silhouette Images** and **Gait Energy Images (GEI)** through a **Channel Attention-based Multimodal Fusion Module**, allowing the network to simultaneously learn spatial body structure and temporal gait dynamics.

The framework has been trained and evaluated on the **CASIA-B gait dataset**, demonstrating robust cross-view gait recognition under multiple walking conditions including:

- Normal Walking (NM)
- Bag Carrying (BG)
- Coat Wearing (CL)

The complete pipeline covers:

- Video preprocessing
- Silhouette extraction
- GEI generation
- Multimodal feature fusion
- Deep feature embedding
- Rank-1 gait identification
- Open-set gait verification using ROC and EER analysis

---

## ✨ Key Features

- ✅ Modified GaitSet architecture with Multimodal Feature Fusion
- ✅ Dual-input learning using Silhouette Images and Gait Energy Images (GEI)
- ✅ Channel Attention Mechanism for adaptive feature weighting
- ✅ Complete preprocessing pipeline for the CASIA-B dataset
- ✅ Cross-view gait recognition
- ✅ Rank-1 Identification using Cosine Similarity
- ✅ Biometric verification using ROC-AUC and Equal Error Rate (EER)
- ✅ GPU-accelerated implementation using PyTorch
- ✅ Modular codebase for easy experimentation and future research

---

# 🏗️ Proposed System Architecture

The proposed framework extends the original **GaitSet** architecture by integrating **Multimodal Feature Fusion** using **Silhouette Images** and **Gait Energy Images (GEI)**. A Channel Attention mechanism is employed to adaptively weight complementary features before generating the final gait embedding for identification and verification.

<p align="center">
  <img src="docs/architecture.png" width="950"/>
</p>

<p align="center">
<b>Figure 1.</b> Complete architecture of the proposed Modified GaitSet-based Multimodal Gait Recognition Framework.
</p>


---

# 📂 Repository Structure

```text
Gait-Multi-Modal-Fusion
│
├── docs/                     # Architecture and result images used in the README
│
├── GaitDatasetB-silh/         # Original CASIA-B silhouette dataset
│
├── Processed_CASIAB/          # Preprocessed dataset (.npy files)
│
├── results/                   # Evaluation results, plots and trained outputs
│
├── baseline_results/          # Baseline experiment results
│
├── gait_env/                  # Python virtual environment (optional)
│
├── preprocess.py              # Data preprocessing pipeline
├── pack_npy.py                # Converts processed data into NumPy format
├── dataset.py                 # CASIA-B dataset loader
├── model.py                   # Modified GaitSet architecture
├── train.py                   # Model training
├── eval.py                    # Rank-1 evaluation
├── compute_biometrics.py      # ROC, AUC and EER evaluation
├── plot_view_matrix.py        # Cross-view accuracy visualization
│
├── requirements.txt
└── README.md
```

