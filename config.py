"""
Configuration for Cognitive Radiology Report Generation Model
HI-CliTr Inspired Architecture with PRO-FA, MIX-MLP, and RCTA modules
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset paths
    data_dir: str = "./data"
    iu_xray_dir: str = "./data/iu_xray"
    mimic_cxr_dir: str = "./data/mimic_cxr"  # Optional - requires PhysioNet access
    
    # Image settings
    image_size: int = 224
    num_channels: int = 3
    
    # Text settings
    max_seq_length: int = 256
    max_findings_length: int = 200
    max_impression_length: int = 100
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Preprocessing
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class EncoderConfig:
    """PRO-FA Visual Encoder Configuration"""
    # Backbone
    backbone: str = "swin_base_patch4_window7_224"  # Swin-Transformer-Base
    pretrained: bool = True
    
    # Multi-scale feature dimensions
    pixel_dim: int = 1024   # Stage 4 output
    region_dim: int = 512   # Stage 3 output  
    organ_dim: int = 256    # Stage 2 output
    
    # Feature map sizes (for 224x224 input)
    pixel_size: int = 7     # 7x7 feature map
    region_size: int = 14   # 14x14 feature map
    organ_size: int = 28    # 28x28 feature map (will be pooled)
    
    # RadLex alignment
    radlex_embed_dim: int = 512
    num_radlex_concepts: int = 50  # Core anatomical concepts
    
    # Output
    hidden_dim: int = 768
    dropout: float = 0.1


@dataclass
class ClassifierConfig:
    """MIX-MLP Classifier Configuration"""
    # Input
    input_dim: int = 768
    
    # Multi-path MLP
    residual_dim: int = 512
    expansion_ratio: int = 4  # Hidden layer expansion
    num_hidden_layers: int = 2
    
    # CheXpert labels (14 pathologies)
    num_labels: int = 14
    chexpert_labels: List[str] = field(default_factory=lambda: [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ])
    
    # Loss
    pos_weight: float = 2.0  # For class imbalance
    label_smoothing: float = 0.1
    
    # Dropout
    dropout: float = 0.2


@dataclass
class DecoderConfig:
    """RCTA + Report Generator Configuration"""
    # RCTA Attention
    hidden_dim: int = 768
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    
    # Triangular attention stages
    num_rcta_layers: int = 3
    
    # Report Generator (GPT-2)
    generator_model: str = "gpt2-medium"  # Can use BioMedLM if available
    vocab_size: int = 50257
    max_length: int = 256
    
    # Generation settings
    beam_size: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    
    # Dropout
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 2
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    
    # Training duration
    max_epochs: int = 30
    early_stopping_patience: int = 5
    
    # Mixed precision
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    
    # Multi-task loss weights
    lambda_cls: float = 1.0      # Classification loss weight
    lambda_gen: float = 1.0      # Generation loss weight
    lambda_align: float = 0.5    # Alignment loss weight
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 0.5  # Validate twice per epoch
    
    # Reproducibility
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation Configuration"""
    # Metrics
    use_chexpert_f1: bool = True
    use_radgraph_f1: bool = True
    use_nlg_metrics: bool = True
    
    # NLG metrics
    nlg_metrics: List[str] = field(default_factory=lambda: [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
        "METEOR", "ROUGE-L", "CIDEr"
    ])
    
    # Weights (from competition)
    weight_chexpert: float = 0.4
    weight_radgraph: float = 0.3
    weight_nlg: float = 0.3


@dataclass
class ModelConfig:
    """Complete Model Configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Model name
    model_name: str = "HiCliTr-CogRad"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)


# Default configuration instance
def get_config() -> ModelConfig:
    """Get default model configuration"""
    return ModelConfig()


# RadLex core anatomical concepts for chest X-ray
RADLEX_CONCEPTS = [
    # Organs
    "heart", "lung", "pleura", "mediastinum", "diaphragm", "trachea",
    "aorta", "pulmonary artery", "esophagus", "thyroid",
    
    # Lung regions
    "right upper lobe", "right middle lobe", "right lower lobe",
    "left upper lobe", "left lower lobe", "lingula",
    "right hilum", "left hilum", "carina",
    
    # Heart regions
    "left ventricle", "right ventricle", "left atrium", "right atrium",
    "aortic arch", "descending aorta", "ascending aorta",
    
    # Bones
    "rib", "clavicle", "scapula", "spine", "sternum",
    
    # Other structures
    "costophrenic angle", "cardiophrenic angle", "hemidiaphragm",
    "pulmonary vasculature", "bronchus", "fissure",
    
    # Common pathology locations
    "apex", "base", "perihilar", "peripheral", "central"
]


# CheXpert uncertainty mappings
CHEXPERT_UNCERTAINTY_POLICIES = {
    "zeros": 0,       # Treat uncertain as negative
    "ones": 1,        # Treat uncertain as positive
    "ignore": -1,     # Ignore uncertain labels
    "self_train": 2,  # Use for self-training
}
