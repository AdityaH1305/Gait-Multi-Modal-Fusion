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