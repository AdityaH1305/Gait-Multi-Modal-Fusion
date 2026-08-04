# Cross-View Gait Recognition via Multimodal Fusion

### A Deep Learning Pipeline for Human Identification Combining a Modified GaitSet Backbone with Multimodal Feature Fusion

---

## Introduction

Gait is a behavioral biometric — it identifies people by how they walk rather than by a static physical trait. Because it can be captured from a distance and does not require the subject's cooperation, unlike face or fingerprint recognition, gait analysis is well suited to surveillance, forensics, and security applications.

This repository contains a full implementation of a multimodal gait recognition framework built on a modified GaitSet backbone, based on the paper *"Research on Gait Recognition Based on GaitSet and Multimodal Fusion."*

Rather than relying on silhouettes alone, the system fuses Silhouette Images with Gait Energy Images (GEI) through a channel attention-based fusion module, allowing the network to capture both the spatial structure of the body and the temporal rhythm of the gait cycle.

The pipeline was trained and evaluated on the CASIA-B benchmark across three walking conditions:

- Normal walking (NM)
- Carrying a bag (BG)
- Wearing a coat (CL)

End to end, the project covers video preprocessing, silhouette extraction, GEI generation, multimodal feature fusion, deep embedding learning, Rank-1 identification, and open-set verification (ROC / EER).

---

## Key Features

- Modified GaitSet backbone with multimodal fusion
- Joint learning from silhouette images and GEI
- Channel attention for adaptive feature weighting
- End-to-end CASIA-B preprocessing pipeline
- Cross-view recognition support
- Cosine-similarity-based Rank-1 identification
- Verification via ROC-AUC and Equal Error Rate (EER)
- PyTorch implementation, GPU-accelerated
- Modular codebase for experimentation

---

## Architecture

The framework builds on GaitSet by adding a multimodal feature fusion stage that combines silhouette images with gait energy images. A channel attention module adaptively weights the two feature streams before they are merged into a single gait embedding, which is used for both identification and verification.

<p align="center">
  <img src="docs/architecture.png" width="450">
</p>

<p align="center">
<b>Figure 1.</b> End-to-end pipeline of the modified GaitSet multimodal framework.
</p>

---

## Repository Layout

```text
Gait-Multi-Modal-Fusion
│
├── docs/                     # Images used in this README (architecture, results)
│
├── GaitDatasetB-silh/         # Raw CASIA-B silhouettes
│
├── Processed_CASIAB/          # Preprocessed data (.npy)
│
├── results/                   # Evaluation outputs and plots
│
├── baseline_results/          # Baseline experiment outputs
│
├── gait_env/                  # Optional virtual environment
│
├── preprocess.py              # Preprocessing pipeline
├── pack_npy.py                # Converts processed frames to .npy
├── dataset.py                 # CASIA-B dataset loader
├── model.py                   # Modified GaitSet model definition
├── train.py                   # Training script
├── eval.py                    # Rank-1 evaluation
├── compute_biometrics.py      # ROC / AUC / EER computation
├── plot_view_matrix.py        # Cross-view accuracy heatmap
│
├── requirements.txt
└── README.md
```

---

## Background

Gait recognition identifies people from the pattern of their walk. Because it works at a distance and does not require cooperation, it has drawn sustained interest for surveillance, border control, forensics, and smart-city applications.

The original GaitSet model treated a walking sequence as an unordered set of silhouette frames, a design that pushed cross-view recognition forward considerably. Its main weakness is that it depends entirely on silhouette shape, which makes it sensitive to anything that changes the outline of the body, such as heavy coats or carried bags.

This project addresses that limitation with a modified GaitSet architecture, following the approach described in *"Research on Gait Recognition Based on GaitSet and Multimodal Fusion."* Instead of a single silhouette stream, the model fuses silhouette images with gait energy images (GEI) through a channel attention-based fusion module, so the network learns from:

- Spatial body shape, from silhouettes
- Temporal walking dynamics, from GEI
- An adaptively weighted combination of the two, via channel attention

The resulting embeddings are more discriminative and more resilient to appearance changes than a silhouette-only baseline.

---

## Dataset

Training and evaluation were carried out on CASIA-B, one of the standard benchmarks for cross-view gait recognition.

| Property | Value |
|----------|-------|
| Dataset | CASIA-B |
| Subjects | 124 |
| Camera views | 11 (0°–180°) |
| Walking conditions | Normal (NM), Bag (BG), Coat (CL) |
| Gallery subjects | 75–124 |
| Probe sequences | NM, BG, CL |
| Gallery sequences | NM-01 to NM-04 |

CASIA-B's range of viewpoints and appearance conditions makes it a demanding, and informative, testbed for this type of model.

---

## Getting Started

To reproduce the results or train the model from scratch, see the full setup walkthrough, covering installation, environment configuration, dataset preparation, and training.

**[Getting Started Guide](docs/GETTING_STARTED.md)**

---

## Results & Performance

This section covers how the model trained, how accurately it identifies subjects, how well it performs at verification, and what its attention mechanism learned over the course of training.

### Summary

| Metric | Result |
|:--------|:------:|
| Overall Rank-1 accuracy | 75.20% |
| Normal walking (NM) | 98.00% |
| Bag carrying (BG) | 82.24% |
| Coat wearing (CL) | 45.36% |
| ROC AUC | 0.5876 |
| Equal Error Rate (EER) | 44.94% |

The model performs strongly under normal walking conditions and holds up reasonably well when a bag is introduced. Performance under heavy clothing is the clear weak point. The verification metrics offer a complementary view of how well the embeddings separate genuine samples from impostors.

### Training Behavior

Training was tracked using cross-entropy loss, batch-all triplet loss, and training accuracy over 150 epochs.

<p align="center">
  <img src="docs/training_curves.png" width="850">
</p>

<p align="center">
<b>Figure 2.</b> Cross-entropy loss, batch-all triplet loss, and training accuracy over 150 epochs.
</p>

Cross-entropy loss falls steadily, reflecting improving classification ability. Batch-all triplet loss drops substantially, showing that embeddings for the same identity are pulled together while different identities are pushed apart. Training accuracy rises consistently across epochs. Overall, training converges smoothly, with no signs of instability in the optimization.

### Identification Results

Rank-1 accuracy, the percentage of probes correctly matched to their identity as the top candidate, was measured separately for NM, BG, and CL conditions.

<p align="center">
  <img src="docs/rank1_table.png" width="700">
</p>

<p align="center">
<b>Figure 3.</b> Rank-1 accuracy on CASIA-B across walking conditions.
</p>

Normal walking achieved the best result, at 98.00%, reflecting the case where appearance variation is minimal and the model can rely on clean gait signal. Bag carrying dropped to 82.24%, a meaningful decline from NM but one the fusion strategy handles well given the moderate appearance change involved. Coat wearing fell to 45.36%, by far the largest drop. Heavy clothing substantially alters the silhouette, and this remains the hardest condition for gait recognition in general; multimodal fusion helps but does not fully resolve it.

The aggregate Rank-1 accuracy of 75.20% reflects strong performance on the easier conditions offset by the difficulty of the CL scenario.

### Verification Results

Beyond identification, the framework was evaluated on its ability to distinguish genuine matches from impostors.

<p align="center">
  <img src="docs/roc_curve.png" width="700">
</p>

<p align="center">
<b>Figure 4.</b> ROC curve for the verification task.
</p>

| Metric | Value |
|:-------|------:|
| AUC | 0.5876 |
| EER | 44.94% |
| Optimal threshold | 0.0094 |

An AUC of 0.5876 indicates only moderate separability between genuine and impostor pairs. An EER of 44.94%, the point at which false acceptance and false rejection rates are equal, is fairly high, meaning verification is noticeably weaker than identification for this model. The threshold of 0.0094 is the cosine-similarity cutoff that best balances the two error types. Taken together, these numbers suggest the framework is currently better suited to closed-set Rank-1 identification than to open-set verification, though the embeddings still carry useful discriminative signal.

### Attention Visualization

Attention maps were captured at several points during training to observe how the channel attention module's focus evolved.

<p align="center">
  <img src="docs/attention_maps.png" width="600">
</p>

<p align="center">
<b>Figure 5.</b> Attention evolution at epochs 10, 50, 100, and 150 (warm colors indicate higher importance, cool colors indicate lower importance).
</p>

At epoch 10, attention is diffuse, and the model has not yet zeroed in on informative regions. By epoch 50, structure begins to emerge around body regions relevant to gait motion. By epoch 100, focus sharpens further, with background regions increasingly suppressed. By epoch 150, attention settles into a stable, well-localized pattern.

This progression, from broad exploration to targeted focus, tracks with the steady gains seen in the training curves and reflects successful convergence of the fusion mechanism.

---

## Limitations

**Sensitivity to clothing.** CL accuracy (45.36%) lags well behind NM and BG, since heavy or loose clothing obscures the silhouette that the model partly relies on.

**Dependence on silhouette quality.** The pipeline's output is only as good as the silhouette extraction step; segmentation noise or incomplete masks degrade downstream features.

**Single-dataset evaluation.** All results reported here are on CASIA-B; generalization to other gait datasets has not been tested.

**Controlled-capture assumption.** The current setup does not account for poor lighting, occlusion, crowding, or uneven terrain, all of which are common in real-world deployments.

---

## Future Work

- Clothing-invariant feature extraction to close the CL performance gap
- Skeleton- or pose-based features alongside silhouette and GEI for complementary motion cues
- Transformer-based or other advanced attention mechanisms for fusion
- Cross-dataset evaluation to test generalization beyond CASIA-B
- Adaptation to real-world surveillance conditions, including lighting, clutter, and occlusion
- Model compression and inference optimization for edge deployment

---

## Acknowledgements

This project builds on prior open research and public datasets:

- **CASIA-B Gait Dataset**, the benchmark dataset used throughout this work.
- **[GaitSet: Cross-View Gait Recognition Through Utilizing Gait As a Deep Set](https://ieeexplore.ieee.org/document/9351667)**, the set-based representation this implementation extends.
- **[Research on Gait Recognition Based on GaitSet and Multimodal Fusion](https://ieeexplore.ieee.org/document/10852208)**, the source of the multimodal attention-fusion approach used here.
- **PyTorch** and its open-source community, on which the implementation is built.

Thanks also to the faculty and mentors who supported this project's development.
