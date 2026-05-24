# End-to-End Lightweight Spatiotemporal Learning for Practical Human Action Classification

[![Journal-The Visual Computer](https://img.shields.io/badge/Journal-The%20Visual%20Computer-blue.svg)](https://link.springer.com/)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-DOI%20%5BZenodo%20DOI%5D-blue.svg)](https://zenodo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official open-source repository for the paper **"End-to-End Lightweight Spatiotemporal Learning for Practical Human Action Classification"** revised and resubmitted to **The Visual Computer (Springer Nature)**.

This repository provides a unified visual computing framework spanning cloud APIs, mobile edge devices, interactive desktop training gates, and cloud notebook clusters.

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
* *For launch steps and CUDA GPU configuration, open the [Desktop-App README](./Desktop-App/README.md).*

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
│   ├── app/src/main/           # Android UI layouts & camera overlay activities
│   └── apk/                    # Pinned Android binary package (LUMAT.apk)
├── Desktop-App/                # standalone Desktop GUI Control Center (CustomTkinter)
│   ├── gui/                    # Sidebar layout frames and background thread services
│   ├── har/                    # Core deep learning Torch package (config, models, loaders)
│   ├── results/                # checkpoints (300k, 292k, R3D-18), metrics & TensorBoard curves
│   ├── Sample-Test/            # Folder to drop video clips for local GUI testing
│   ├── images/                 # User guide illustrated screenshots
│   ├── run_gui.bat             # Windows one-click auto-setup batch script
│   ├── run_gui.sh              # Linux/macOS one-click auto-setup bash script
│   └── gui.py                  # Entrypoint to run desktop app
├── Google-Colab/               # Cloud GPU training Jupyter notebooks (.ipynb)
│   ├── UCF101_HAR_Pipeline.ipynb
│   └── Kinetics700_HAR_Pipeline.ipynb
├── Web-api/                    # Localhost-configured Flask REST API server
│   ├── app.py                  # API endpoints, frame decoding & predictions
│   ├── model/                  # Standard weights (.pth) & Mobile weights (.ptl)
│   ├── metrics/                # Benchmark JSON accuracy reports
│   └── static/ & templates/    # Web dashboards & Vanilla JS controller
├── requirements.txt            # Python global packages list
├── CITATION.cff                # Standard citation file
└── .gitignore                  # Git repository exclusion settings
```

---

## ⚙️ Global Environment Quickstart

To install Python dependencies globally:

```bash
# Clone the repository
git clone https://github.com/IstiyakV/End-to-End-Lightweight-Spatiotemporal-Learning-for-Practical-Human-Action-Classification.git
cd End-to-End-Lightweight-Spatiotemporal-Learning-for-Practical-Human-Action-Classification

# Install all machine learning, web, and GUI dependencies
pip install -r requirements.txt
```
*Note: For Windows desktop users, double-clicking `Desktop-App/run_gui.bat` automatically isolates packages in a virtual environment (`env`) without polluting your global Python path.*

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
