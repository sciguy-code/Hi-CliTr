"""
MIX-MLP: Multi-path Knowledge-Enhanced Classification Module

Implements dual-path MLP architecture for multi-label CheXpert classification:
- Residual Path: Direct projection with skip connections
- Expansion Path: MLP with hidden expansion for complex patterns
- Label Correlation: Modeling disease co-occurrence

Handles CheXpert's uncertain labels with asymmetric loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config


class ResidualMLP(nn.Module):
    """
    Residual path for direct feature transformation
    with skip connections for gradient flow
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            ))
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        return self.output_proj(x)


class ExpansionMLP(nn.Module):
    """
    Expansion path with hidden layer expansion
    for capturing complex non-linear patterns
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        expansion_ratio: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        expanded_dim = hidden_dim * expansion_ratio
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(expanded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LabelCorrelationModule(nn.Module):
    """
    Models disease co-occurrence patterns using
    learnable label correlation matrix
    """
    
    def __init__(
        self,
        num_labels: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Learnable label embeddings
        self.label_embed = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.02)
        
        # Co-occurrence attention
        self.correlation_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, label_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            label_features: Per-label features [B, num_labels, D]
            
        Returns:
            Correlation-adjusted logits [B, num_labels]
        """
        B = label_features.shape[0]
        
        # Add learnable label embeddings
        label_emb = self.label_embed.unsqueeze(0).expand(B, -1, -1)
        label_features = label_features + label_emb
        
        # Cross-label attention for co-occurrence
        correlated, _ = self.correlation_attn(
            label_features, label_features, label_features
        )
        
        # Residual
        label_features = label_features + correlated
        
        # Project to logits
        logits = self.output_proj(label_features).squeeze(-1)  # [B, num_labels]
        
        return logits


class MIXMLPClassifier(nn.Module):
    """
    Complete MIX-MLP Module for Multi-label Classification
    
    Architecture:
    - Residual Path: Efficient direct transformation
    - Expansion Path: Complex pattern modeling  
    - Label Correlation: Disease co-occurrence
    
    Output: 14 CheXpert pathology probabilities
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_labels: int = 14,
        expansion_ratio: int = 4,
        num_residual_layers: int = 2,
        dropout: float = 0.2,
        use_label_correlation: bool = True,
    ):
        super().__init__()
        
        config = get_config()
        
        self.num_labels = num_labels
        self.label_names = config.classifier.chexpert_labels
        self.use_label_correlation = use_label_correlation
        self.input_dim = input_dim  # Store for label embedding output
        self.hidden_dim = hidden_dim
        
        # Feature transformation
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Dual-path classification
        self.residual_path = ResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim if use_label_correlation else num_labels,
            num_layers=num_residual_layers,
            dropout=dropout,
        )
        
        self.expansion_path = ExpansionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim if use_label_correlation else num_labels,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
        )
        
        # Path fusion
        if use_label_correlation:
            # Per-label projection
            self.label_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.label_correlation = LabelCorrelationModule(num_labels, hidden_dim)
            
            # Additional per-label heads
            self.label_heads = nn.ModuleList([
                nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_labels)
            ])
            
            # Project label embeddings to full input_dim for decoder compatibility
            self.label_embed_proj = nn.Linear(hidden_dim, input_dim)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(num_labels * 2, num_labels * 2),
                nn.LayerNorm(num_labels * 2),
                nn.GELU(),
                nn.Linear(num_labels * 2, num_labels),
            )
        
    def forward(
        self,
        features: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Visual features [B, D]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with:
            - logits: Classification logits [B, num_labels]
            - probabilities: Sigmoid probabilities [B, num_labels]
            - embeddings: Optional intermediate features
        """
        B = features.shape[0]
        
        # Normalize input
        features = self.input_norm(features)
        
        # Dual-path forward
        residual_out = self.residual_path(features)
        expansion_out = self.expansion_path(features)
        
        if self.use_label_correlation:
            # Concatenate paths
            combined = torch.cat([residual_out, expansion_out], dim=-1)  # [B, hidden*2]
            
            # Per-label features
            label_features = []
            for head in self.label_heads:
                label_feat = head(combined)  # [B, hidden]
                label_features.append(label_feat)
            
            label_features = torch.stack(label_features, dim=1)  # [B, num_labels, hidden]
            
            # Apply label correlation
            logits = self.label_correlation(label_features)  # [B, num_labels]
            
            embeddings = label_features
        else:
            # Direct fusion
            combined = torch.cat([residual_out, expansion_out], dim=-1)
            logits = self.fusion(combined)
            embeddings = None
        
        # Probabilities
        probabilities = torch.sigmoid(logits)
        
        output = {
            "logits": logits,
            "probabilities": probabilities,
        }
        
        if return_embeddings and embeddings is not None:
            # Project to full input_dim for decoder compatibility
            embeddings = self.label_embed_proj(embeddings)  # [B, num_labels, input_dim]
            output["label_embeddings"] = embeddings
        
        return output
    
    def get_predictions(
        self,
        logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Get binary predictions from logits"""
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).float()
        
        return {
            "predictions": predictions,
            "probabilities": probs,
        }


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-label Classification
    
    Handles:
    - Class imbalance (more negatives than positives)
    - CheXpert uncertain labels
    - Hard example mining via gamma focusing
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [B, num_labels]
            targets: Ground truth labels [B, num_labels] (0, 1, or -1 for uncertain)
            mask: Optional mask for valid labels
            
        Returns:
            Loss value
        """
        # Probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Basic CE
        los_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        
        loss = los_pos + los_neg
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            pt0 = probs_pos * targets
            pt1 = probs_neg * (1 - targets)
            pt = pt0 + pt1
            
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            loss *= one_sided_w
        
        # Handle uncertain labels (-1)
        if mask is None:
            # Create mask for valid (non-uncertain) labels
            mask = (targets >= 0).float()
        
        loss = loss * mask
        
        # Average over valid labels
        return -loss.sum() / (mask.sum() + self.eps)


class UncertaintyAwareLoss(nn.Module):
    """
    CheXpert-specific loss handling uncertain labels
    
    Strategies:
    - zeros: Treat uncertain as negative (U-Zeros)
    - ones: Treat uncertain as positive (U-Ones)
    - ignore: Ignore uncertain labels
    - self-train: Use model predictions for uncertain
    """
    
    def __init__(
        self,
        strategy: str = "zeros",
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.strategy = strategy
        self.label_smoothing = label_smoothing
        
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        
        self.asymmetric_loss = AsymmetricLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_labels]
            targets: [B, num_labels] with values in {-1, 0, 1}
                    -1 = uncertain, 0 = negative, 1 = positive
        """
        # Handle uncertain labels based on strategy
        processed_targets = targets.clone()
        uncertain_mask = (targets == -1)
        
        if self.strategy == "zeros":
            processed_targets[uncertain_mask] = 0
        elif self.strategy == "ones":
            processed_targets[uncertain_mask] = 1
        elif self.strategy == "ignore":
            pass  # Will be masked out
        elif self.strategy == "self_train":
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                processed_targets[uncertain_mask] = (probs[uncertain_mask] > 0.5).float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            processed_targets = processed_targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Create mask
        if self.strategy == "ignore":
            mask = (~uncertain_mask).float()
        else:
            mask = None
        
        return self.asymmetric_loss(logits, processed_targets, mask)


if __name__ == "__main__":
    # Test the classifier
    print("Testing MIX-MLP Classifier...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create classifier
    classifier = MIXMLPClassifier(
        input_dim=768,
        hidden_dim=512,
        num_labels=14,
        use_label_correlation=True,
    ).to(device)
    
    # Test forward pass
    features = torch.randn(4, 768).to(device)
    output = classifier(features, return_embeddings=True)
    
    print(f"\nInput features shape: {features.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probabilities shape: {output['probabilities'].shape}")
    print(f"Label embeddings shape: {output['label_embeddings'].shape}")
    
    # Test loss
    loss_fn = UncertaintyAwareLoss(strategy="zeros")
    targets = torch.randint(-1, 2, (4, 14)).float().to(device)
    loss = loss_fn(output["logits"], targets)
    
    print(f"\nTargets shape: {targets.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nMIX-MLP Classifier test passed! ✓")
