"""
Complete Cognitive Radiology Model

Integrates all three mandatory modules:
- PRO-FA: Hierarchical Visual Perception with RadLex alignment
- MIX-MLP: Knowledge-Enhanced Multi-label Classification  
- RCTA: Triangular Cognitive Attention with Report Generation

This is the main model class for the Hi-CliTr inspired architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models.encoder import PROFAEncoder
from models.classifier import MIXMLPClassifier, UncertaintyAwareLoss
from models.decoder import RCTADecoder


class CognitiveRadiologyModel(nn.Module):
    """
    Hi-CliTr Inspired Cognitive Radiology Report Generation Model
    
    Architecture:
    1. PRO-FA Encoder: Multi-scale visual features aligned with RadLex
    2. MIX-MLP Classifier: 14-class CheXpert pathology prediction
    3. RCTA Decoder: Triangular cognitive attention + GPT-2 generation
    
    The model simulates radiologist cognitive workflow:
    - Hierarchical visual perception (whole → region → pixel)
    - Mental diagnosis formation (classification)
    - Verification loop before report generation
    """
    
    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        hidden_dim: int = 768,
        num_labels: int = 14,
        num_heads: int = 12,
        num_rcta_layers: int = 3,
        generator_model: str = "gpt2",
        dropout: float = 0.1,
        freeze_encoder_stages: int = 0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # PRO-FA: Hierarchical Visual Encoder with RadLex alignment
        self.encoder = PROFAEncoder(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # MIX-MLP: Multi-path Classification
        self.classifier = MIXMLPClassifier(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_labels=num_labels,
            use_label_correlation=True,
            dropout=dropout,
        )
        
        # RCTA + Generator
        self.decoder = RCTADecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_rcta_layers=num_rcta_layers,
            pretrained_generator=generator_model,
            dropout=dropout,
        )

        # Safety projection for label embeddings when classifier uses a smaller hidden_dim
        self.label_align = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # Tokenizer for generation
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def forward(
        self,
        images: torch.Tensor,
        indication_ids: torch.Tensor,
        indication_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        chexpert_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the cognitive pipeline
        
        Args:
            images: CXR images [B, V, 3, H, W] or [B, 3, H, W]
            indication_ids: Clinical indication tokens [B, N_t]
            indication_mask: Indication attention mask [B, N_t]
            input_ids: Target report tokens [B, L] (for training)
            attention_mask: Target attention mask [B, L]
            labels: Target labels for generation loss [B, L]
            chexpert_labels: Ground truth CheXpert labels [B, 14]
            
        Returns:
            Dictionary containing:
            - visual_features: PRO-FA output
            - classification_logits: MIX-MLP logits
            - classification_probs: Sigmoid probabilities
            - generation_logits: Text generation logits
            - loss_cls: Classification loss
            - loss_gen: Generation loss
            - total_loss: Combined loss
        """
        # Stage 1: PRO-FA Visual Encoding
        encoder_output = self.encoder(images)
        visual_features = encoder_output["visual_features"]  # [B, N, D]
        pooled_features = encoder_output["pooled_features"]  # [B, D]
        
        # Stage 2: MIX-MLP Classification
        cls_output = self.classifier(pooled_features, return_embeddings=True)
        classification_logits = cls_output["logits"]  # [B, 14]
        classification_probs = cls_output["probabilities"]
        label_embeddings = cls_output["label_embeddings"]  # [B, 14, D]

        # Ensure label embeddings match visual feature dimension
        if label_embeddings.size(-1) != visual_features.size(-1):
            label_embeddings = self.label_align(label_embeddings)
        
        result = {
            "visual_features": visual_features,
            "pooled_features": pooled_features,
            "classification_logits": classification_logits,
            "classification_probs": classification_probs,
            "label_embeddings": label_embeddings,
        }
        
        # Stage 3: RCTA + Generation (if training)
        if input_ids is not None:
            decoder_output = self.decoder(
                visual_features=visual_features,
                label_embeddings=label_embeddings,
                indication_ids=indication_ids,
                indication_mask=indication_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            result["verified_features"] = decoder_output["verified_features"]
            result["generation_logits"] = decoder_output["logits"]
            result["loss_gen"] = decoder_output["loss"]
        
        # Compute classification loss
        if chexpert_labels is not None:
            loss_fn = UncertaintyAwareLoss(strategy="zeros")
            result["loss_cls"] = loss_fn(classification_logits, chexpert_labels)
        
        # Combined loss
        if "loss_cls" in result and "loss_gen" in result:
            config = get_config()
            result["total_loss"] = (
                config.training.lambda_cls * result["loss_cls"] +
                config.training.lambda_gen * result["loss_gen"]
            )
        elif "loss_cls" in result:
            result["total_loss"] = result["loss_cls"]
        elif "loss_gen" in result:
            result["total_loss"] = result["loss_gen"]
        
        return result
    
    @torch.no_grad()
    def generate_report(
        self,
        images: torch.Tensor,
        indication: Union[str, torch.Tensor, None] = None,
        max_findings_length: int = 150,
        max_impression_length: int = 75,
        return_labels: bool = True,
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        """
        Generate complete radiology report from image(s)
        
        Args:
            images: CXR images [B, V, 3, H, W] or [B, 3, H, W]
            indication: Optional clinical indication (string or tokens)
            max_findings_length: Max tokens for findings
            max_impression_length: Max tokens for impression
            return_labels: Whether to return predicted CheXpert labels
            
        Returns:
            Dictionary with:
            - findings: List of findings texts
            - impressions: List of impression texts
            - reports: Complete reports (findings + impression)
            - predicted_labels: Optional CheXpert predictions
        """
        device = images.device
        B = images.shape[0]
        
        # Handle indication
        if indication is None:
            indication = "Clinical indication not provided."
        
        if isinstance(indication, str):
            # Tokenize
            enc = self.tokenizer(
                indication,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            indication_ids = enc["input_ids"].expand(B, -1).to(device)
            indication_mask = enc["attention_mask"].expand(B, -1).to(device)
        else:
            indication_ids = indication
            indication_mask = torch.ones_like(indication)
        
        # PRO-FA encoding
        encoder_output = self.encoder(images)
        visual_features = encoder_output["visual_features"]
        pooled_features = encoder_output["pooled_features"]
        
        # MIX-MLP classification
        cls_output = self.classifier(pooled_features, return_embeddings=True)
        label_embeddings = cls_output["label_embeddings"]
        predicted_labels = cls_output["probabilities"]

        # Ensure label embeddings match visual feature dimension
        if label_embeddings.size(-1) != visual_features.size(-1):
            label_embeddings = self.label_align(label_embeddings)
        
        # RCTA + Generation
        reports = self.decoder.generate_report(
            visual_features=visual_features,
            label_embeddings=label_embeddings,
            indication_ids=indication_ids,
            indication_mask=indication_mask,
            tokenizer=self.tokenizer,
            max_findings_length=max_findings_length,
            max_impression_length=max_impression_length,
        )
        
        # Format complete reports
        complete_reports = []
        for i in range(B):
            report = f"FINDINGS: {reports['findings'][i]}\n\nIMPRESSION: {reports['impressions'][i]}"
            complete_reports.append(report)
        
        result = {
            "findings": reports["findings"],
            "impressions": reports["impressions"],
            "reports": complete_reports,
        }
        
        if return_labels:
            result["predicted_labels"] = predicted_labels
            result["label_names"] = self.classifier.label_names
        
        return result


def create_model(
    config=None,
    pretrained: bool = True,
    device: str = "cuda",
) -> CognitiveRadiologyModel:
    """
    Factory function to create the model
    
    Args:
        config: Optional model configuration
        pretrained: Whether to use pretrained weights
        device: Device to place model on
        
    Returns:
        Initialized CognitiveRadiologyModel
    """
    if config is None:
        config = get_config()
    
    model = CognitiveRadiologyModel(
        backbone=config.encoder.backbone,
        pretrained=pretrained,
        hidden_dim=config.encoder.hidden_dim,
        num_labels=config.classifier.num_labels,
        num_heads=config.decoder.num_attention_heads,
        num_rcta_layers=config.decoder.num_rcta_layers,
        generator_model=config.decoder.generator_model,
        dropout=config.encoder.dropout,
    )
    
    return model.to(device)


if __name__ == "__main__":
    # Test the complete model
    print("=" * 60)
    print("Testing Complete Cognitive Radiology Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(pretrained=True, device=device)
    
    # Test inputs
    B = 2
    images = torch.randn(B, 2, 3, 224, 224).to(device)  # Multi-view
    indication_ids = torch.randint(0, 1000, (B, 32)).to(device)
    indication_mask = torch.ones(B, 32).to(device)
    input_ids = torch.randint(0, 1000, (B, 64)).to(device)
    attention_mask = torch.ones(B, 64).to(device)
    labels = input_ids.clone()
    chexpert_labels = torch.randint(0, 2, (B, 14)).float().to(device)
    
    # Forward pass
    print("\nRunning forward pass...")
    output = model(
        images=images,
        indication_ids=indication_ids,
        indication_mask=indication_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        chexpert_labels=chexpert_labels,
    )
    
    print(f"\nForward pass results:")
    print(f"  Visual features: {output['visual_features'].shape}")
    print(f"  Classification logits: {output['classification_logits'].shape}")
    print(f"  Generation logits: {output['generation_logits'].shape}")
    print(f"  Classification loss: {output['loss_cls'].item():.4f}")
    print(f"  Generation loss: {output['loss_gen'].item():.4f}")
    print(f"  Total loss: {output['total_loss'].item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    print("\n" + "=" * 60)
    print("Complete Model test passed! ✓")
    print("=" * 60)
