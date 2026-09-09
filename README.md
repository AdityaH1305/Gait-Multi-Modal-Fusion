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

### Evaluation Protocol

Protocol choices move these numbers substantially, so they are stated up front. Gallery and probe sets come from subjects 075–124, which are disjoint from every identity seen during training.

| Item | Setting |
|:-----|:--------|
| Training identities | 001–064 |
| Validation identities | 065–074, held out for checkpoint selection |
| Test identities | 075–124, never seen during training or model selection |
| Gallery | `nm-01`–`nm-04`, averaged into one template per (subject, angle) — 550 templates |
| Probes | NM: `nm-05`, `nm-06` · BG: `bg-01`, `bg-02` · CL: `cl-01`, `cl-02` |
| Matching | Cosine similarity on 256-D L2-normalised embeddings |
| Frames | All frames per sequence; Set Pooling is invariant to sequence length |
| Cross-view averaging | All 11 × 11 gallery/probe angle pairs, excluding the identical-view diagonal |
| Verification pairing | Every probe against every template, identical-view pairs excluded |

Matching a probe against gallery footage from the *same* camera angle is a considerably easier problem than genuine cross-view recognition, so the diagonal is excluded from the headline figures. Both are reported below, since the gap between them is itself informative.

Evaluation is deterministic for a fixed `--frame-budget`, though changing it can shift individual figures by up to about 0.1 point: batch shapes change, cuDNN selects different convolution algorithms, and the resulting floating-point differences flip a small number of near-tied nearest-neighbour decisions. Only CL is measurably affected, as it has the narrowest genuine/impostor margin. All figures here use the default budget of 768.

### Summary

Figures below are for the reference model: the current pipeline (aspect-preserving preprocessing) trained with occlusion augmentation.

| Metric | Cross-view | Same-view |
|:-------|:----------:|:---------:|
| Overall Rank-1 accuracy | **65.99%** | 94.03% |
| Normal walking (NM) | **78.31%** | 99.73% |
| Bag carrying (BG) | **69.06%** | 97.63% |
| Coat wearing (CL) | **50.62%** | 84.73% |

| Verification | AUC | EER | Threshold |
|:-------------|:---:|:---:|:---------:|
| Normal walking (NM) | **0.9707** | **9.21%** | 0.2796 |
| Bag carrying (BG) | **0.9555** | **11.38%** | 0.2513 |
| Coat wearing (CL) | **0.9231** | **15.71%** | 0.2009 |
| Combined | **0.9486** | **12.38%** | 0.2390 |

The model identifies subjects reliably under normal walking and degrades predictably as appearance changes: a carried bag costs roughly 9 points of Rank-1 accuracy, heavy clothing around 28. Verification is markedly stronger than identification, which is expected — deciding whether a single pair matches is an easier problem than selecting the right identity from 50 candidates. Clothing remains the weak point, though the gap has narrowed considerably.

> **One configuration scores higher on identification.** Training with occlusion augmentation on the *previous* preprocessing reaches **69.07%** overall cross-view Rank-1 (NM 82.04 / BG 72.34 / CL 52.83), about 3 points above the reference model, while scoring lower on verification (AUC 0.9401, EER 13.47%). The two changes do not stack, and the reason is not yet established — see [Development History](#development-history) below. The reference model is the one on the corrected preprocessing, since that pipeline is the maintained one.

### Training Behavior

Training is tracked using cross-entropy loss, batch-hard triplet loss, training accuracy, and cross-view Rank-1 on the held-out validation identities, over 150 epochs.

<p align="center">
  <img src="docs/training_curves.png" width="1000">
</p>

<p align="center">
<b>Figure 2.</b> Cross-entropy loss, batch-hard triplet loss, training accuracy, and validation cross-view Rank-1. The dashed line on the first panel marks ln(64), the loss of a uniform random guess.
</p>

Cross-entropy falls from the random-guess level to near zero, and the batch-hard triplet loss drops alongside it, showing that same-identity embeddings are pulled together while different identities are pushed apart.

Training accuracy reaches essentially 100% within the first few dozen epochs. That is the important reading: **the model is not optimisation-limited, it is data-limited.** Once training accuracy saturates, further epochs cannot improve the fit to the training identities, and everything that matters is happening in the generalisation gap. This is why the fourth panel exists — validation Rank-1 on held-out identities is the only curve that distinguishes genuine progress from memorisation, and it is what selects the saved checkpoint.

An earlier revision of this section claimed training "converges smoothly, with no signs of instability." That was measured on a run that never converged at all; see [Corrections](#corrections-to-previously-reported-results).

### Identification Results

Rank-1 accuracy, the percentage of probes correctly matched to their identity as the top candidate, was measured separately for NM, BG, and CL conditions. The full 11 × 11 matrix is reported rather than a single averaged figure, since performance depends heavily on the angle between the gallery and probe cameras.

<p align="center">
  <img src="docs/cross_view_matrix.png" width="980">
</p>

<p align="center">
<b>Figure 3.</b> Cross-view Rank-1 accuracy for every gallery/probe angle pair. Rows are gallery angles, columns are probe angles. The red-outlined diagonal is same-view matching and is excluded from the cross-view mean.
</p>

Normal walking achieved the best result, at 78.31%, reflecting the case where appearance variation is minimal and the model can rely on clean gait signal. Bag carrying reached 69.06%, a modest decline given that a carried object corrupts part of the silhouette while leaving the underlying gait dynamics intact. Coat wearing fell to 50.62%, still the largest drop. Heavy clothing substantially alters the silhouette, and because the architecture collapses the fused feature map into a single global embedding, a coat contaminates the whole descriptor rather than only the affected body region — which is the motivation for the part-based head listed under [Future Work](#future-work).

The aggregate cross-view accuracy of 65.99% reflects solid performance on the easier conditions offset by the continuing difficulty of the CL scenario.

Three patterns in the matrix are worth noting, all physically meaningful rather than artefacts of the measurement. Accuracy falls as the gallery/probe angle gap widens — 94.3% for gaps within two steps, 73.4% for gaps of six or more — the expected signature of a view-dependent representation. The 0° and 180° probe columns are weakest (68.5% against 78.6% for the mid-range views), since a subject walking directly toward or away from the camera shows far less lateral limb motion. And pairs at opposite extremes recover sharply: gallery 0° against probe 180° reaches 98% under NM, because those two viewpoints are near-mirror images of the same walking motion.

### Verification Results

Beyond identification, the framework was evaluated on its ability to distinguish genuine matches from impostors. This is the open-set question relevant to deployment: whether a *single* global similarity threshold can accept genuine pairs and reject impostors across identities the model has never seen.

<p align="center">
  <img src="docs/roc_curve.png" width="700">
</p>

<p align="center">
<b>Figure 4.</b> ROC curves per walking condition and combined, computed from real gallery and probe embeddings with identical-view pairs excluded.
</p>

| Probe | AUC | EER | Threshold | Genuine mean | Impostor mean |
|:------|----:|----:|----------:|-------------:|--------------:|
| NM | 0.9707 | 9.21% | 0.2796 | +0.5485 | +0.0459 |
| BG | 0.9555 | 11.38% | 0.2513 | +0.4803 | +0.0483 |
| CL | 0.9231 | 15.71% | 0.2009 | +0.3854 | +0.0340 |
| Combined | 0.9486 | 12.38% | 0.2390 | +0.4714 | +0.0427 |

A combined AUC of 0.9486 indicates the embedding space separates genuine from impostor pairs well, rising to 0.9707 under normal walking alone. The corresponding EER — the point at which false acceptance and false rejection rates are equal — is 12.38% combined and 9.21% for NM.

The threshold is worth reading carefully, because it does not transfer across conditions. It falls from 0.28 under NM to 0.20 under CL as genuine-pair similarity degrades, so a deployed system running one fixed cutoff would either admit impostors under NM or turn away genuine users under CL. Reporting a single threshold without stating the condition it was tuned on would be misleading, which is why all four appear above.

Absolute similarity values are not comparable across model revisions — the CosFace head reshapes the angular geometry of the embedding space, so thresholds here sit lower than in earlier revisions while separating the two distributions better. What matters is the gap between the genuine and impostor means, not either value alone.

<p align="center">
  <img src="docs/score_distributions.png" width="980">
</p>

<p align="center">
<b>Figure 5.</b> Genuine and impostor cosine similarity distributions per condition. The overlap between the two is what sets the achievable EER.
</p>

Figure 5 shows why CL is the hardest case, and the mechanism is more specific than "clothing hurts." The impostor distribution barely moves across conditions, sitting near +0.04 throughout. What changes is the genuine distribution, which slides down and broadens from +0.55 under NM to +0.39 under CL. The failure is a loss of similarity between genuine pairs, not an increase in false matches — which points at feature extraction rather than at the decision rule.

### Attention Visualization

Attention maps were captured at several points during training to observe how the channel attention module's focus evolved.

<p align="center">
  <img src="docs/attention_maps.png" width="600">
</p>

<p align="center">
<b>Figure 6.</b> Attention evolution at epochs 10, 50, 100, and 150 (warm colors indicate higher importance, cool colors indicate lower importance).
</p>

At epoch 10, attention is diffuse, and the model has not yet zeroed in on informative regions. By epoch 50, structure begins to emerge around body regions relevant to gait motion. By epoch 100, focus sharpens further, with background regions increasingly suppressed. By epoch 150, attention settles into a stable, well-localized pattern.

This progression, from broad exploration to targeted focus, tracks with the steady gains seen in the training curves and reflects successful convergence of the fusion mechanism.

---

## Limitations

**Sensitivity to clothing.** CL accuracy (50.62%) still lags NM and BG, since heavy or loose clothing obscures the silhouette that the model partly relies on. The single global embedding compounds this: because the fused feature map is flattened and projected into one vector, a coat degrades the entire descriptor rather than only the body region it covers.

**Overfitting is the binding constraint.** Training accuracy reaches 100% within a few dozen epochs on 64 identities, and roughly 85% of the model's 8.6M parameters sit in the single fully-connected layer that projects the flattened feature map. Longer training cannot help; only more data diversity, stronger regularisation, or a smaller head can.

**Preprocessing and occlusion augmentation do not compose.** Each helps on its own but together they underperform occlusion alone, and the interaction is not yet understood. See [Development History](#development-history).

**Verification thresholds are condition-specific.** The optimal cutoff varies from 0.28 (NM) to 0.20 (CL), so a single fixed threshold would not serve all conditions equally.

**Dependence on silhouette quality.** The pipeline's output is only as good as the silhouette extraction step; segmentation noise or incomplete masks degrade downstream features. A handful of CASIA-B test sequences contain as few as two usable frames, and these are evaluated as-is rather than excluded.

**Single-dataset evaluation.** All results reported here are on CASIA-B; generalization to other gait datasets has not been tested.

**Controlled-capture assumption.** The current setup does not account for poor lighting, occlusion, crowding, or uneven terrain, all of which are common in real-world deployments.

---

## Development History

Every change below was measured in isolation on the held-out test identities, one at a time, so each row's delta is attributable. Overall cross-view Rank-1:

| Run | Data | Occlusion | Change from previous row | NM | BG | CL | **Overall** | Δ |
|:----|:----:|:---------:|:-------------------------|---:|---:|---:|------------:|---:|
| — | v1 | – | Starting point, correct protocol | 71.24 | 57.65 | 31.18 | **53.36** | — |
| A | v1 | – | Epoch defined by sequences, larger PK batch | 77.61 | 62.41 | 38.88 | **59.63** | +6.27 |
| B | v1 | – | CosFace head | 77.05 | 63.77 | 41.15 | **60.66** | +1.03 |
| C | v1 | – | LR 1e-4 with warmup | 72.11 | 55.67 | 34.95 | **54.24** | −6.42 |
| D | v1 | 0.3 | 400 epochs, occlusion (on top of C) | 70.22 | 56.27 | 40.35 | **55.61** | +1.37 |
| E | v1 | 0.3 | Occlusion on top of **B** instead of C | 82.04 | 72.34 | 52.83 | **69.07** | +8.41 |
| F | **v2** | – | Aspect-preserving preprocessing, from B | 79.59 | 66.87 | 44.75 | **63.74** | +3.08 |
| **F2** | **v2** | 0.3 | Preprocessing **and** occlusion | 78.31 | 69.06 | 50.62 | **65.99** | −3.08 |

Four findings are worth recording.

**Training-loop correctness dominated everything else.** Run A is a single change with no new architecture: the batch sampler previously defined an epoch as one pass over *identities* rather than sequences, yielding 18 batches per epoch and about 1,350 optimizer steps across an entire 150-epoch run. Cross-entropy never left the random-guess region. Fixing the epoch definition was worth more than every architectural change combined.

**A plausible hyperparameter change cost 6.4 points.** Run C lowered the learning rate to 1e-4 with warmup, a conventional and defensible choice. It was worse at every validation checkpoint. Because the ablation ladder is cumulative, run D inherited that damage — which is why run D's occlusion augmentation appeared to be worth almost nothing. Re-testing the same augmentation on the healthy configuration (run E) showed it was worth **+8.41**, the second-largest gain in the project. Ablating one change at a time is what surfaced this; a single combined run would have buried it.

**Preprocessing helps, but only in the absence of occlusion augmentation.** Fixing the aspect-ratio distortion is worth +3.08 without occlusion (B → F) and −3.08 with it (E → F2). The symmetry is striking and the cause is not established. The plausible reading is that both changes address the same weakness — over-reliance on precise silhouette geometry — so their benefits overlap, and an occlusion rate tuned implicitly against noisy data over-regularises once the data is clean. Re-tuning the occlusion probability on the corrected pipeline is the obvious next experiment.

**Identification and verification do not rank the models identically.** Run E leads on cross-view Rank-1 (69.07% against 65.99%), while F2 leads on verification (AUC 0.9486 against 0.9401, EER 12.38% against 13.47%) and on same-view accuracy (94.03% against 92.72%). Which model is "best" depends on whether the application ranks candidates or thresholds pairs.

A methodological caveat: runs A–E were measured on the pre-fix preprocessing. Conclusions that depend on optimizer step count (run A) transfer to the corrected data with confidence; the smaller effects, particularly the CosFace margin, have not been re-verified on it.

---

## Future Work

- Re-tune the occlusion probability on the corrected preprocessing, to resolve the interaction recorded above
- Horizontal Pyramid Mapping in place of the single flattened embedding, so that a coat degrades only the affected body strips rather than the whole descriptor — this also cuts roughly 85% of the parameters, targeting the overfitting directly
- A 64×44 input crop, the GaitSet convention, since the aspect-preserving silhouette now occupies only about 24 of 64 columns
- Clothing-invariant feature extraction to close the CL performance gap, including entropy-based representations that encode only the moving regions of the body
- A genuinely complementary second modality: the GEI is the pixel-wise mean of the same frames the dynamic branch already receives, so the fusion module currently weighs two views of one signal
- Skeleton- or pose-based features alongside silhouette and GEI for complementary motion cues
- Transformer-based or other advanced attention mechanisms for fusion
- Cross-dataset evaluation to test generalization beyond CASIA-B
- Adaptation to real-world surveillance conditions, including lighting, clutter, and occlusion
- Model compression and inference optimization for edge deployment

---

## Corrections to Previously Reported Results

An earlier revision of this README reported figures that were produced incorrectly. They have been replaced with measured values, and the causes are recorded here.

**Identification used a same-view protocol.** Each probe was matched only against gallery sequences recorded from its own camera angle, which is same-view rather than cross-view recognition. The previously reported NM 98.00% / BG 82.24% / CL 45.36% correspond to the diagonal of Figure 3 and remain reproducible as the same-view column in the Summary. The cross-view figures now reported follow the standard protocol with that diagonal excluded.

**Verification metrics were computed from random data.** `compute_biometrics.py` contained a test harness that generated `np.random.randn` vectors, and the scoring function was never called with real embeddings. The previously reported AUC 0.5876 / EER 44.94% / threshold 0.0094 were that harness's output; a near-chance AUC paired with a near-zero threshold is the signature of unrelated random vectors rather than of this model. The measured values are substantially better than the figures they replace.

**The cross-view matrix was generated by a formula.** `plot_view_matrix.py` filled each cell with `92.0 - (angle_distance * 6.5)` plus seeded noise and never loaded the model. Figure 3 is now computed from real embeddings.

**Training was described as converging when it never did.** The claim that training "converges smoothly, with no signs of instability" was made about a run that completed roughly 1,350 optimizer steps in total, because the batch sampler defined an epoch as one pass over identities rather than sequences. Cross-entropy started at ln(74) = 4.30 — exactly the random-guess value — and had only reached 3.12 after 150 epochs, still falling linearly. The curves were smooth because the run stopped long before anything interesting happened.

**Preprocessing was distorting the data rather than normalising it.** `crop_and_resize` cropped each silhouette to its bounding box and resized it to a square. Measured across 720 frames spanning every subject, condition and angle, the raw CASIA-B silhouettes are *already* 64×64, already span the full height, and are already horizontally centred to within a pixel. The step therefore normalised nothing and did nothing but stretch the width — by a median of 2.67× and up to 4.57×, varying 2.47× from frame to frame within a single sequence as the limbs moved. Set Pooling was taking a maximum over frames whose geometry disagreed, and the GEI was averaging bodies at inconsistent widths; roughly 45% of GEI pixels were ambiguous mid-grey blur, against 14% after the fix.

Several changes were made to prevent this class of error from recurring:

- `eval.py` is the only script that loads the model. It performs a single GPU pass and writes `results/embeddings.npz`, which the reporting scripts consume rather than constructing their own inputs.
- Before reporting anything, `eval.py` asserts embeddings are unit-norm and prints mean genuine against mean impostor similarity, warning explicitly if the two are indistinguishable.
- `compute_biometrics.py` emits the score distributions in Figure 5 alongside the ROC, where a degenerate result is immediately visible rather than hidden inside a summary statistic.
- `train.py` refuses to start if the dataset is missing identities or the validation split is empty, rather than silently training on a fraction of the data. An interrupted preprocessing run had previously produced a model fitted to 21 identities instead of 64, with no validation and therefore no saved checkpoint.
- Training writes a resumable checkpoint every epoch, written atomically so an interrupt cannot leave a truncated file, and refuses to resume if any run-defining argument has changed.
- `preprocess.py --compare` renders a before/after figure so a normalisation change can be inspected on a handful of sequences before committing to reprocessing the dataset.

The broader lesson is that each of these errors came from trusting a description of what code did instead of measuring what it produced. The preprocessing bug in particular survived because the function was named `crop_and_resize` and appeared to normalise; only measuring the raw data revealed there was nothing left to normalise.

---

## Acknowledgements

This project builds on prior open research and public datasets:

- **CASIA-B Gait Dataset**, the benchmark dataset used throughout this work.
- **[GaitSet: Cross-View Gait Recognition Through Utilizing Gait As a Deep Set](https://ieeexplore.ieee.org/document/9351667)**, the set-based representation this implementation extends.
- **[Research on Gait Recognition Based on GaitSet and Multimodal Fusion](https://ieeexplore.ieee.org/document/10852208)**, the source of the multimodal attention-fusion approach used here.
- **PyTorch** and its open-source community, on which the implementation is built.

Thanks also to the faculty and mentors who supported this project's development.
