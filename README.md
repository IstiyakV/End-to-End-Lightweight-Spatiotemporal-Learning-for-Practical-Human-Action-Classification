# End-to-End Lightweight Spatiotemporal Learning for Practical Human Action Classification

<div align="center">

[![Journal-The Visual Computer](https://img.shields.io/badge/Journal-The%20Visual%20Computer-blue.svg)](https://link.springer.com/)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-DOI%20%5BZenodo%20DOI%5D-blue.svg)](https://zenodo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Official open-source repository for the paper "End-to-End Lightweight Spatiotemporal Learning for Practical Human Action Classification" revised and resubmitted to The Visual Computer (Springer Nature).**

This repository provides a unified visual computing framework spanning cloud APIs, mobile edge devices, interactive desktop training gates, and cloud notebook clusters.

<br>

<img src="Desktop-App/images/demo-playing.gif" width="700" alt="HAR Control Center Live Demo" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">

<br>

<a href="#-global-environment-quickstart">
  <img src="https://img.shields.io/badge/%E2%9A%A1%20Quick%20Installation-Click%20Here-blueviolet?style=for-the-badge&logo=quick" alt="Quick Installation">
</a>

</div>

---

## 📑 Table of Contents
- [The Four-Folder Repository Architecture](#-the-four-folder-repository-architecture)
- [Codebase Directory Layout](#-codebase-directory-layout)
- [Application Screenshots & Interface Guide](#-application-screenshots--interface-guide)
- [System Requirements & Prerequisites](#-system-requirements--prerequisites)
- [Global Environment Quickstart](#-global-environment-quickstart)
- [Dataset Ingestion & Official Downloads](#-dataset-ingestion--official-downloads)
- [Model Training & Testing Guide](#-model-training--testing-guide)
- [Real-Time YouTube Video Inference](#-real-time-youtube-video-inference)
- [How to Register a Permanent Zenodo DOI](#-how-to-register-a-permanent-zenodo-doi)
- [Citation](#-citation)

---

## 🚀 The Four-Folder Repository Architecture

To ensure structured navigation for researchers, developers, and peer-reviewers, this repository is organized into exactly **four primary folders**:

### 1. [📂 Desktop-App](./Desktop-App)
The local research workstation module containing the Tkinter-based interactive GUI app (`HAR Control Center`).
* **Visual Graph Builder:** Construct custom convolutional network layers and trace dimensions.
* **Stateful Monitor:** Training loop supervisor featuring live pause, resume, and history-safe retraining.
* **Baseline Suite:** Preloaded with the three model weights discussed in the paper:
  1. `R(2+1)D-Light 300k` (303,576 parameters, 82.06% accuracy on UCF101)
  2. `Plain 3D CNN 292k` (292,325 parameters baseline)
  3. `R3D-18 Backbone` (Kinetics pre-trained weights)
* **Logs & Metrics:** Complete TensorBoard run event logs and accuracy metrics reports.
* **Auto-Launchers:** One-click double-click scripts (`run_gui.bat` / `run_gui.sh`) that set up virtual environments and install libraries.

### 2. [📂 Web-api](./Web-api)
The production Gunicorn/Flask cloud REST API and interactive web dashboard.
* **Decoupled Stream Processing:** Multi-threaded query queue, uniform frame segment normalization, and high-speed Grad-CAM attention visualizers.
* **Subdomain Cleanup:** Configured out-of-the-box to run on `http://localhost:5000/` for local testing.
* **Mobile Compilation Weight Assets:** Includes the PyTorch Mobile Lite (`.ptl`) offline weight checkpoints so mobile clients can download them on-device.

### 3. [📂 Android](./Android)
The native Gradle-based Android Studio mobile edge client.
* **Camera Overlay HUD:** Operates a real-time camera stream overlay HUD processing sub-20ms frame classification queries.
* **Conversational Helper:** Chat UI interface integrating action predictions into help assistance dialogs.
* **Compiled Package:** Includes `apk/LUMAT.apk` ready for direct mobile installation.

### 4. [📂 Google-Colab](./Google-Colab)
Polished Jupyter Notebook training pipelines pre-configured to run on cloud T4 GPUs:
* `UCF101_HAR_Pipeline.ipynb`: UCF101 uniform segment download, pre-processing, training, and Grad-CAM exporting.
* `Kinetics700_HAR_Pipeline.ipynb`: Decoupled asynchronous segmented downsampling manager to train on the Kinetics dataset.

---

## 📂 Codebase Directory Layout

```text
.
├── Android/                    # Native Android Studio project (Java)
├── Desktop-App/                # Standalone Desktop GUI Control Center
│   ├── gui/                    # Sidebar layout frames and background thread services
│   ├── har/                    # Core deep learning Torch package (config, models, loaders)
│   ├── results/                # Checkpoints (300k, 292k, R3D-18), metrics & curves
│   ├── Sample-Test/            # Folder to drop video clips for local GUI testing
│   ├── images/                 # User guide illustrated screenshots
│   ├── run_gui.bat             # Windows one-click auto-setup batch script
│   └── run_gui.sh              # Linux/macOS one-click auto-setup bash script
├── Google-Colab/               # Cloud GPU training Jupyter notebooks (.ipynb)
├── Web-api/                    # Localhost-configured Flask REST API server
├── CITATION.cff                # Standard citation file
└── .gitignore                  # Git repository exclusion settings
```

---

## 🎨 Application Screenshots & Interface Guide

Our **HAR Control Center** (Desktop-App) provides an interactive deep learning research workbench. Here is a glimpse of the powerful visual features you can use without writing any code:

| Research Dashboard | Network Architect |
| :---: | :---: |
| <img src="Desktop-App/images/dashboard.png" width="400" alt="Dashboard"> | <img src="Desktop-App/images/network_architect.png" width="400" alt="Network Architect"> |
| Real-time project status, metrics, and precision/recall charts. | Visually compile custom 3D neural topologies from scratch. |

| Transfer Learning | Model Tester |
| :---: | :---: |
| <img src="Desktop-App/images/transfer_learning.png" width="400" alt="Transfer Learning"> | <img src="Desktop-App/images/model_tester.png" width="400" alt="Model Tester"> |
| Fine-tune backbones with stateful control and live loss graphs. | Test local videos or YouTube streams with live Grad-CAM analysis. |

*For more detailed feature breakdowns and all 8 graphical panels, please read the [Desktop-App README](./Desktop-App/README.md).*

---

## 📋 System Requirements & Prerequisites

Before running the application, make sure your system satisfies the following hardware and software specifications:

### 1. Operating System Compatibility
* **Windows**: Windows 10 or 11 (64-bit)
* **Linux**: Ubuntu 20.04 LTS, 22.04 LTS, or newer derivatives
* **macOS**: macOS 11 Big Sur or newer (supports Intel and Apple Silicon)

### 2. Python Environment (Mandatory)
The application requires **Python 3.9, 3.10, or 3.11** (Python 3.10/3.11 recommended).
* **Download:** [Official Python Downloads](https://www.python.org/downloads/)
* **Windows Tip:** Make sure to check **"Add Python.exe to PATH"** during installation.

### 3. Hardware Requirements
* **Memory (RAM):** 8 GB minimum (16 GB recommended).
* **Storage Space:** 500 MB for app files, +2 GB for datasets/checkpoints.
* **CPU:** Multicore Intel/AMD x86_64 or Apple M-series (AVX2 supported).
* **GPU (Recommended):** NVIDIA GPU with Tensor Cores (4GB+ VRAM). CUDA toolkit and GPU-enabled PyTorch highly recommended for mixed-precision (FP16) speedups.

---

<div id="-global-environment-quickstart"></div>

## ⚙️ Global Environment Quickstart

You can easily set up the project locally. 

```bash
# 1. Clone the repository
git clone https://github.com/IstiyakV/End-to-End-Lightweight-Spatiotemporal-Learning-for-Practical-Human-Action-Classification.git
cd End-to-End-Lightweight-Spatiotemporal-Learning-for-Practical-Human-Action-Classification

# 2. Enter the Desktop-App directory
cd Desktop-App
```

**For Windows Users:**
Simply double-click the `run_gui.bat` file! It will automatically create a virtual environment (`env`), install all PyTorch and GUI dependencies, and launch the application.

**For Linux/macOS Users:**
```bash
chmod +x run_gui.sh
./run_gui.sh
```

*(Alternatively, you can manually install the required packages using `pip install -r requirements.txt` inside your preferred Python environment).*

---

## 📊 Dataset Ingestion & Official Downloads

To train or evaluate our compact models, download the official spatiotemporal video benchmarks from the following verified archives:

* **UCF101 Dataset:**
  * **Official Portal:** [UCF101 Action Recognition](https://www.crcv.ucf.edu/data/UCF101.php)
  * **Direct Dataset Download:** [UCF101.rar (~6.5 GB)](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
  * **Direct Class Annotations:** [UCF101 Train/Test Splits](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)
* **Kinetics-700 Dataset:**
  * **Official Portal:** [Kinetics Dataset on DeepMind](https://github.com/google-deepmind/kinetics-dataset)
  * **Consolidated Hugging Face Host:** [Kinetics-700 HF Dataset Archive](https://huggingface.co/datasets/atalaydenknalbant/Kinetics-700)
  
*Note: Our custom multi-threaded downloader inside `Google-Colab/Kinetics700_HAR_Pipeline.ipynb` is pre-configured to stochastically fetch and downscale these Kinetics-700 archives on-the-fly to bound your local SSD footprint.*

---

## 🏋️ Model Training & Testing Guide

You can train and evaluate the spatiotemporal classifiers using either the local Python CLI scripts or the interactive **HAR Control Center** GUI app.

### 1. Training from Scratch
To start a PyTorch training session for our factorised `R(2+1)D-Light` (300k parameter) network on the UCF101 dataset:
```bash
cd Desktop-App
python -m har.train --model r21d_light --epochs 100 --batch_size 8 --lr 0.0001
```

### 2. Evaluating Model Accuracy & Generalization
To run a strict evaluation pass against the validation split, outputting Top-K accuracies, precision-recall metrics, and a confusion matrix:
```bash
cd Desktop-App
python -m har.evaluate --model r21d_light --weight results/checkpoints/ucf101_paper_x112_b16_l2_d30_300k_best.pth
```

### 3. Spatiotemporal Visual Interpretability (Grad-CAM)
To compute and export Grad-CAM spatiotemporal activation heatmap overlays explaining the model's focus during motion:
```bash
cd Desktop-App
python -m har.gradcam --video Sample-Test/action.mp4 --weight results/checkpoints/ucf101_paper_x112_b16_l2_d30_300k_best.pth
```

---

## 📺 Real-Time YouTube Video Inference

Our system enables serverless, high-speed testing of live YouTube streaming clips. The backend processes streams on-the-fly by parsing metadata without downloading raw video files. 

### Launch Local Web Server:
```bash
cd Web-api
python app.py
```

### Stream Live YouTube Videos:
1. Open your web browser and navigate to `http://localhost:5000/`.
2. Scroll to the **YouTube Video Streaming** panel on the web interface.
3. Paste any public YouTube link (e.g. `https://www.youtube.com/watch?v=xxxx`) into the text bar.
4. Select your active model and click **Resolve and Stream**.
5. The backend leverages `yt-dlp` to capture direct stream URLs, performs frame extraction, runs inference, and renders **spatiotemporal Grad-CAM heatmap overlay animations** directly in your browser.

**Quick Testing URLs:**
You can test the system with the following YouTube videos (also available in the Desktop App with 1-click clipboard copy):
* `https://www.youtube.com/watch?v=wOEKdWrtz6U`
* `https://www.youtube.com/watch?v=wIYD42DV3Ro`
* `https://www.youtube.com/watch?v=msXtQTh81jA`
* `https://www.youtube.com/watch?v=EnBQcffEKLc`
* `https://www.youtube.com/watch?v=zVqvd6mhat8`
* `https://www.youtube.com/watch?v=wEVAlMTeyWc`

---

## 📝 How to Register a Permanent Zenodo DOI

To register a permanent scientific DOI for this repository upon paper acceptance:
1. Log in to [zenodo.org](https://zenodo.org/) using your **GitHub account**.
2. Navigate to your Zenodo profile GitHub settings and toggle sync **On** for this repository.
3. Publish a new GitHub release (e.g. `v1.0.0`). Zenodo will automatically archive the code, compile details, and assign a permanent DOI. Replace `[Zenodo DOI]` inside the paper manuscript and files.

---

## 🎓 Citation

Please cite our journal publication in your research:

```bibtex
@article{human_action_classification_2026,
  author    = {The Authors},
  title     = {End-to-End Lightweight Spatiotemporal Learning for Practical Human Action Classification},
  journal   = {The Visual Computer},
  publisher = {Springer Nature},
  year      = {2026},
  note      = {Submitted for Publication (Revised Resubmission)}
}
```

---
*Developed by the Authors for The Visual Computer journal submission.*
