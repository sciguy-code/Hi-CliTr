# 🩺 Hi-CliTr: Cognitive Radiology Report Generation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Soumadipta-Konar/Hi-CliTr/graphs/commit-activity)

> A state-of-the-art Deep Learning framework for automated chest X-ray report generation implementing **Cognitive Simulation** inspired by the Hi-CliTr framework.

---

## 📖 Overview

**Hi-CliTr** (Hierarchical Cross-modal Cognitive Transformer) is designed to address "reader fatigue" in radiology by acting as an intelligent "Second Reader". Unlike standard image captioning models, Hi-CliTr simulates the cognitive workflow of a radiologist:

1.  **Perceives** anatomical structures at multiple scales (Organ → Region → Pixel).
2.  **Reasons** about potential pathologies using a knowledge graph.
3.  **Verifies** its findings against the image before generating the final report.

This project implements the core components: **PRO-FA** (Progressive Feature Alignment), **MIX-MLP** (Knowledge-Enhanced Classification), and **RCTA** (Triangular Cognitive Attention).

---

## 🌟 Key Features

### 1. [PRO-FA (Progressive Feature Alignment)](#pro-fa)
Implements **Hierarchical Visual Perception** via a **Swin-Transformer** backbone. It aligns multi-scale features with the **RadLex** medical ontology:
*   **Organ-level (4×4)**: Global anatomical awareness.
*   **Region-level (7×7)**: Lobe/region specific features.
*   **Pixel-level (7×7)**: Fine-grained lesion details.

### 2. [MIX-MLP (Multi-path Classifier)](#mix-mlp)
A dual-path, knowledge-enhanced architecture for disease classification:
*   **Residual Path**: Efficient feature flow for common cases.
*   **Expansion Path**: Captures complex disease patterns and co-occurrences.
*   **CheXpert**: High-precision classification for **14 common pathologies**.

### 3. [RCTA (Triangular Cognitive Attention)](#rcta)
A 3-stage **closed-loop verification** system that mimics clinical reasoning:
1.  **Image → Text**: Creates context from visual features and clinical indication.
2.  **Context → Labels**: Formulates a diagnostic hypothesis.
3.  **Labels → Image**: Verifies the hypothesis against visual evidence.

### 4. [GPT-2 Generator](#generator)
Utilizes a **GPT-2 Medium** backbone (355M params) to generate structured, clinically accurate reports (Findings & Impression), conditioned on the cognitive states from RCTA.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE RADIOLOGY MODEL                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   CXR       │    │  Clinical   │    │                         │  │
│  │  Images     │    │ Indication  │    │     Generated Report    │  │
│  │  (PA/LAT)   │    │   Text      │    │  ┌──────────────────┐  │  │
│  └──────┬──────┘    └──────┬──────┘    │  │ FINDINGS:        │  │  │
│         │                  │           │  │ Heart size normal│  │  │
│         ▼                  │           │  │ Lungs are clear  │  │  │
│  ┌─────────────────────┐   │           │  ├──────────────────┤  │  │
│  │      PRO-FA         │   │           │  │ IMPRESSION:      │  │  │
│  │ ┌─────┬─────┬─────┐ │   │           │  │ No acute         │  │  │
│  │ │Organ│Regio│Pixel│ │   │           │  │ abnormality      │  │  │
│  │ │ 4x4 │ 7x7 │ 7x7 │ │   │           │  └──────────────────┘  │  │
│  │ └──┬──┴──┬──┴──┬──┘ │   │           │                         │  │
│  │    └─────┼─────┘    │   │           └─────────────────────────┘  │
│  │    RadLex Align     │   │                      ▲                 │
│  └──────────┬──────────┘   │                      │                 │
│             │              │           ┌──────────┴──────────┐      │
│             ▼              │           │   Report Generator  │      │
│  ┌─────────────────────┐   │           │      (GPT-2)        │      │
│  │      MIX-MLP        │   │           └──────────┬──────────┘      │
│  │ ┌─────────────────┐ │   │                      ▲                 │
│  │ │  Residual Path  │ │   │                      │                 │
│  │ ├─────────────────┤ │   │           ┌──────────┴──────────┐      │
│  │ │ Expansion Path  │ │   │           │       RCTA          │      │
│  │ └────────┬────────┘ │   │           │  ┌──────────────┐   │      │
│  │          ▼          │   │           │  │ Image→Text   │   │      │
│  │   14 CheXpert Labels│   └──────────►│  │ Text→Labels  │   │      │
│  │   (Multi-label F1)  │───────────────│  │ Labels→Image │   │      │
│  └─────────────────────┘               │  └──────────────┘   │      │
│                                        └─────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
BrainDead-Solution/
├── data/                       # Data management
│   ├── __init__.py
│   ├── download_iu_xray.py     # Scripts for downloading & preprocessing IU-Xray
│   ├── dataset.py              # PyTorch Dataset definitions (MIMIC-CXR, IU-Xray)
│   └── sanity_check.py         # Data integrity verification script
├── models/                     # Core model components
│   ├── __init__.py
│   ├── encoder.py              # PRO-FA: Multi-scale ViT + RadLex Alignment
│   ├── classifier.py           # MIX-MLP: Knowledge-enhanced classifier
│   ├── decoder.py              # RCTA + GPT-2 Decoder
│   └── model.py                # Unified Hi-CliTr Model assembly
├── training/                   # training logic
│   ├── __init__.py
│   └── trainer.py              # Training loop, validation, and saving
├── evaluation/                 # Metrics and evaluation
│   ├── __init__.py
│   └── metrics.py              # CheXpert F1, BLEU, CIDEr, RadGraph F1
├── notebooks/
│   └── inference_demo.ipynb    # Interactive Jupyter notebook for demo
├── static/                     # Web app static assets (CSS/JS)
├── templates/                  # Web app HTML templates
├── app.py                      # Flask Web Application entry point
├── config.py                   # Centralized configuration file
├── requirements.txt            # Python dependencies
├── problem_statement.md        # Original hackathon problem statement
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/your-username/braindead-solution.git
cd braindead-solution

# Create virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

This project uses the **IU-Xray** dataset for benchmarking and public usage. A helper script is provided to download and prepare it.

```bash
# Verify integrity of existing data or download
python data/download_iu_xray.py --verify

# Preprocess dataset (create splits and metadata)
python data/download_iu_xray.py --preprocess

# Run a sanity check to ensure everything is loadable
python data/sanity_check.py
```

*Note: For **MIMIC-CXR**, you must have credentialed access via PhysioNet. Place the dataset in `data/mimic_cxr` if available.*

### 3. Training

Train the model from scratch using the `trainer.py` script. You can configure hyperparameters in `config.py` or pass them as arguments.

```bash
# Standard training run
python training/trainer.py \
    --max_epochs 30 \
    --batch_size 8 \
    --learning_rate 1e-4

# Fast dev run (sanity check training loop)
python training/trainer.py --fast_dev_run
```

### 4. Web Application (Demo)

Launch the interactive web interface to generate reports for uploaded X-rays.

```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

*   Upload a Chest X-ray image.
*   (Optional) Enter clinical indication (e.g., "Fever and cough").
*   View the generated **Findings** and **Impression**.

### 5. Inference (CLI / Notebook)

You can also run inference programmatically:

```python
from models.model import create_model
import torch

# Load Model
model = create_model(pretrained=True, device="cuda")
model.load_state_dict(torch.load("checkpoints/best.pt")["model_state_dict"])
model.eval()

# Generate Report
result = model.generate_report(
    images="path/to/xray.png", 
    indication="Patient with shortness of breath"
)
print(result['reports'][0])
```

See `notebooks/inference_demo.ipynb` for a complete walkthrough.

---

## ⚙️ Configuration

The `config.py` file controls all aspects of the model and training. Key sections:

*   **`DataConfig`**: Paths, image size (224x224), sequence lengths.
*   **`EncoderConfig`**: Swin Transformer settings, RadLex concept count.
*   **`ClassifierConfig`**: CheXpert labels, loss weights.
*   **`DecoderConfig`**: GPT-2 settings, beam search parameters (k=4).
*   **`TrainingConfig`**: Learning rate, batch size, mixed precision (AMP) settings.

---

## 📊 Performance Targets

| Metric | IU-Xray Test | Target | Description |
|--------|-------------|--------|-------------|
| **CheXpert Micro F1** | *TBD* | **> 0.500** | Clinical accuracy of disease detection |
| **RadGraph F1** | *TBD* | **> 0.500** | Semantic relation accuracy |
| **CIDEr** | *TBD* | **> 0.400** | Text generation consensus metric |
| **BLEU-4** | *TBD* | **> 0.100** | N-gram overlap precision |

---

## 🤝 Contributing

Contributions are welcome!
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{braindead2026,
  title={Hi-CliTr: Cognitive Radiology Report Generation},
  author={Team BrainDead},
  booktitle={ML Hackathon 2026},
  year={2026}
}
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Made with 🧠 and ❤️ by <b>Team BrainDead</b> for <b>ML Hackathon 2026</b>
  <br>
  <i>"Pushing the boundaries of Cognitive Simulation in Radiology"</i>
</p>
