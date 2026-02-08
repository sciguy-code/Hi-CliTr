# 🩺 Hi-CliTr: Cognitive Radiology Report Generation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A state-of-the-art Deep Learning framework for automated chest X-ray report generation implementing **Cognitive Simulation** inspired by the Hi-CliTr framework.

<p align="center">
  <img src="docs/architecture.png" alt="Architecture Diagram" width="800"/>
</p>

## 🌟 Highlights

- **PRO-FA**: Hierarchical Visual Perception with RadLex medical ontology alignment
- **MIX-MLP**: Knowledge-enhanced multi-label classification for 14 CheXpert pathologies
- **RCTA**: Triangular Cognitive Attention with closed-loop verification
- **GPT-2 Generator**: Medical text generation for Findings and Impressions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE RADIOLOGY MODEL                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   CXR       │    │  Clinical   │    │                         │ │
│  │  Images     │    │ Indication  │    │     Generated Report    │ │
│  │  (PA/LAT)   │    │   Text      │    │  ┌──────────────────┐  │ │
│  └──────┬──────┘    └──────┬──────┘    │  │ FINDINGS:        │  │ │
│         │                  │           │  │ Heart size normal│  │ │
│         ▼                  │           │  │ Lungs are clear  │  │ │
│  ┌─────────────────────┐   │           │  ├──────────────────┤  │ │
│  │      PRO-FA         │   │           │  │ IMPRESSION:      │  │ │
│  │ ┌─────┬─────┬─────┐ │   │           │  │ No acute         │  │ │
│  │ │Organ│Regio│Pixel│ │   │           │  │ abnormality      │  │ │
│  │ │ 4x4 │ 7x7 │ 7x7 │ │   │           │  └──────────────────┘  │ │
│  │ └──┬──┴──┬──┴──┬──┘ │   │           │                         │ │
│  │    └─────┼─────┘    │   │           └─────────────────────────┘ │
│  │    RadLex Align     │   │                      ▲                │
│  └──────────┬──────────┘   │                      │                │
│             │              │           ┌──────────┴──────────┐     │
│             ▼              │           │   Report Generator  │     │
│  ┌─────────────────────┐   │           │      (GPT-2)        │     │
│  │      MIX-MLP        │   │           └──────────┬──────────┘     │
│  │ ┌─────────────────┐ │   │                      ▲                │
│  │ │  Residual Path  │ │   │                      │                │
│  │ ├─────────────────┤ │   │           ┌──────────┴──────────┐     │
│  │ │ Expansion Path  │ │   │           │       RCTA          │     │
│  │ └────────┬────────┘ │   │           │  ┌──────────────┐   │     │
│  │          ▼          │   │           │  │ Image→Text   │   │     │
│  │   14 CheXpert Labels│   └──────────►│  │ Text→Labels  │   │     │
│  │   (Multi-label F1)  │───────────────│  │ Labels→Image │   │     │
│  └─────────────────────┘               │  └──────────────┘   │     │
│                                        └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
BrainDead-Solution/
├── data/
│   ├── __init__.py
│   ├── download_iu_xray.py     # Dataset download & preprocessing
│   └── dataset.py              # PyTorch Dataset implementations
├── models/
│   ├── __init__.py
│   ├── encoder.py              # PRO-FA: Multi-scale ViT + RadLex
│   ├── classifier.py           # MIX-MLP: Dual-path classifier
│   ├── decoder.py              # RCTA + GPT-2 generator
│   └── model.py                # Complete unified model
├── training/
│   ├── __init__.py
│   └── trainer.py              # Training pipeline
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # CheXpert F1, NLG metrics
├── notebooks/
│   └── inference_demo.ipynb    # Interactive demo
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/braindead-solution.git
cd braindead-solution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download IU-Xray from Kaggle
python data/download_iu_xray.py --download --preprocess

# Verify dataset
python data/download_iu_xray.py --verify
```

### Training

```bash
# Full training
python training/trainer.py \
    --data_dir ./data/iu_xray \
    --max_epochs 30 \
    --batch_size 8 \
    --learning_rate 1e-4

# Quick test run
python training/trainer.py --fast_dev_run
```

### Inference

```python
from models.model import create_model
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = create_model(pretrained=True, device="cuda")
model.load_state_dict(torch.load("checkpoints/best.pt")["model_state_dict"])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = transform(Image.open("xray.png")).unsqueeze(0).cuda()

# Generate report
with torch.no_grad():
    result = model.generate_report(
        images=image,
        indication="55M with fever and cough"
    )

print(result["reports"][0])
```

## 📊 Performance

| Metric | IU-Xray Test | Target |
|--------|-------------|--------|
| CheXpert Micro F1 | TBD | > 0.500 |
| RadGraph F1 | TBD | > 0.500 |
| CIDEr | TBD | > 0.400 |
| BLEU-4 | TBD | 0.100+ |

## 🧠 Technical Details

### PRO-FA (Progressive Feature Alignment)

- **Backbone**: Swin-Transformer-Base (88M params, ImageNet-22K pretrained)
- **Multi-scale features**:
  - Organ-level: 4×4 (global anatomy)
  - Region-level: 7×7 (lobe/region)
  - Pixel-level: 7×7 (lesion details)
- **RadLex Integration**: 50 core anatomical concepts with learnable embeddings

### MIX-MLP (Multi-path Classifier)

- **Dual-path architecture**:
  - Residual path: Efficient skip-connected transformation
  - Expansion path: 4× hidden expansion for complex patterns
- **Label correlation**: Graph attention for disease co-occurrence
- **Loss**: Asymmetric focal loss for class imbalance

### RCTA (Triangular Cognitive Attention)

- **3-stage verification**:
  1. Image → Clinical Text (context creation)
  2. Context → Labels (hypothesis formation)
  3. Hypothesis → Image (closed-loop verification)
- **L=3 layers** with multi-head cross-attention

### Report Generator

- **Base**: GPT-2 Medium (355M params)
- **Conditioning**: Prefix tokens + cross-attention at layers 2, 5, 8, 11
- **Decoding**: Beam search (k=4) with repetition penalty

## 📝 Citation

```bibtex
@inproceedings{braindead2026,
  title={Hi-CliTr: Cognitive Radiology Report Generation},
  author={Team BrainDead},
  booktitle={ML Hackathon 2026},
  year={2026}
}
```

## 📄 License

This project is for educational and research purposes. MIT License.

---

<p align="center">
  Made with 🧠 by Team BrainDead for ML Hackathon 2026
</p>
