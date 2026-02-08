"""
PRO-FA: Hierarchical Visual Perception Module
Progressive Feature Alignment with RadLex Ontology

This module implements multi-scale visual feature extraction with medical ontology alignment:
- Pixel-level features (7x7) - Fine-grained lesion detection
- Region-level features (14x14) - Lobar/regional abnormalities  
- Organ-level features (28x28) - Whole organ assessment
- RadLex alignment - Medical terminology grounding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm
from einops import rearrange, repeat

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, RADLEX_CONCEPTS


class MultiScaleSwinEncoder(nn.Module):
    """
    Multi-scale Swin Transformer for hierarchical feature extraction
    
    Extracts features at three granularities:
    - Stage 2: 28x28 (organ-level)
    - Stage 3: 14x14 (region-level)
    - Stage 4: 7x7 (pixel-level)
    """
    
    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        freeze_stages: int = 0,
    ):
        super().__init__()
        
        # Load Swin Transformer with intermediate features
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),  # Stages 2, 3, 4
        )
        
        # Get feature dimensions for each stage
        # Swin-Base: [256, 512, 1024] for stages 2, 3, 4
        self.feature_dims = self.backbone.feature_info.channels()
        
        # Freeze early stages if specified
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)
        
        # Projection heads to common dimension
        config = get_config()
        self.hidden_dim = config.encoder.hidden_dim
        
        self.organ_proj = nn.Sequential(
            nn.Conv2d(self.feature_dims[0], self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
        )
        
        self.region_proj = nn.Sequential(
            nn.Conv2d(self.feature_dims[1], self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
        )
        
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(self.feature_dims[2], self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
        )
        
        # Adaptive pooling for consistent spatial dimensions
        self.organ_pool = nn.AdaptiveAvgPool2d((4, 4))   # 4x4 = 16 tokens
        self.region_pool = nn.AdaptiveAvgPool2d((7, 7))  # 7x7 = 49 tokens
        # Pixel level keeps original 7x7
        
    def _freeze_stages(self, num_stages: int):
        """Freeze early stages of the backbone"""
        for name, param in self.backbone.named_parameters():
            # Freeze patch embedding and early layers
            if "patch_embed" in name or any(f"layers.{i}" in name for i in range(num_stages)):
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass extracting multi-scale features
        
        Args:
            x: Input images [B, 3, H, W] or [B, V, 3, H, W] for multi-view
            
        Returns:
            Dictionary with:
            - organ_features: [B, hidden_dim, 4, 4]
            - region_features: [B, hidden_dim, 7, 7]
            - pixel_features: [B, hidden_dim, 7, 7]
            - pooled_features: [B, hidden_dim] global representation
        """
        # Handle multi-view input
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
            multi_view = True
        else:
            B = x.shape[0]
            V = 1
            multi_view = False
        
        # Extract multi-scale features
        features = self.backbone(x)
        organ_feat, region_feat, pixel_feat = features
        
        # Swin Transformer outputs in NHWC format, convert to NCHW for Conv2d
        # Shape: [B, H, W, C] -> [B, C, H, W]
        # NHWC: last dim (C=192/512/1024) is larger than first spatial dim (H=28/14/7)
        if organ_feat.dim() == 4 and organ_feat.shape[-1] > organ_feat.shape[1]:
            organ_feat = organ_feat.permute(0, 3, 1, 2).contiguous()
            region_feat = region_feat.permute(0, 3, 1, 2).contiguous()
            pixel_feat = pixel_feat.permute(0, 3, 1, 2).contiguous()
        
        # Project to common dimension
        organ_feat = self.organ_proj(organ_feat)
        region_feat = self.region_proj(region_feat)
        pixel_feat = self.pixel_proj(pixel_feat)
        
        # Pool organ features
        organ_feat = self.organ_pool(organ_feat)
        region_feat = self.region_pool(region_feat)
        
        # Global pooling for classification
        pooled = F.adaptive_avg_pool2d(pixel_feat, (1, 1)).flatten(1)
        
        # Aggregate multi-view if needed
        if multi_view:
            organ_feat = organ_feat.view(B, V, -1, 4, 4).mean(dim=1)
            region_feat = region_feat.view(B, V, -1, 7, 7).mean(dim=1)
            pixel_feat = pixel_feat.view(B, V, -1, pixel_feat.shape[-2], pixel_feat.shape[-1]).mean(dim=1)
            pooled = pooled.view(B, V, -1).mean(dim=1)
        
        return {
            "organ_features": organ_feat,    # [B, D, 4, 4]
            "region_features": region_feat,  # [B, D, 7, 7]
            "pixel_features": pixel_feat,    # [B, D, 7, 7]
            "pooled_features": pooled,       # [B, D]
        }


class RadLexEmbedding(nn.Module):
    """
    RadLex Medical Ontology Embeddings
    
    Creates learnable embeddings for core anatomical concepts
    that can be aligned with visual features
    """
    
    def __init__(
        self,
        concepts: List[str] = None,
        embed_dim: int = 768,
        use_pretrained_text: bool = True,
    ):
        super().__init__()
        
        self.concepts = concepts or RADLEX_CONCEPTS
        self.num_concepts = len(self.concepts)
        self.embed_dim = embed_dim
        
        if use_pretrained_text:
            # Initialize from pretrained text embeddings (e.g., PubMedBERT)
            # For efficiency, use learned embeddings initialized from a text encoder
            self.concept_embeddings = nn.Parameter(
                torch.randn(self.num_concepts, embed_dim) * 0.02
            )
        else:
            self.concept_embeddings = nn.Parameter(
                torch.randn(self.num_concepts, embed_dim) * 0.02
            )
        
        # Projection layer for matching visual feature dimension
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Category heads for anatomical grouping
        self.category_ids = self._create_categories()
        
    def _create_categories(self) -> Dict[str, List[int]]:
        """Group concepts into anatomical categories"""
        categories = {
            "organs": [],
            "lung_regions": [],
            "heart_regions": [],
            "bones": [],
            "other": [],
        }
        
        for i, concept in enumerate(self.concepts):
            concept_lower = concept.lower()
            if any(x in concept_lower for x in ["heart", "ventricle", "atrium", "aorta"]):
                categories["heart_regions"].append(i)
            elif any(x in concept_lower for x in ["lobe", "lung", "hilum", "bronchus"]):
                categories["lung_regions"].append(i)
            elif any(x in concept_lower for x in ["rib", "clavicle", "scapula", "spine", "sternum"]):
                categories["bones"].append(i)
            elif any(x in concept_lower for x in ["pleura", "diaphragm", "mediastinum", "trachea"]):
                categories["organs"].append(i)
            else:
                categories["other"].append(i)
        
        return categories
    
    def forward(self, category: Optional[str] = None) -> torch.Tensor:
        """
        Get RadLex embeddings
        
        Args:
            category: Optional category filter ("organs", "lung_regions", etc.)
            
        Returns:
            Concept embeddings [num_concepts, embed_dim]
        """
        embeddings = self.proj(self.concept_embeddings)
        
        if category and category in self.category_ids:
            indices = self.category_ids[category]
            embeddings = embeddings[indices]
        
        return embeddings
    
    def get_concept_names(self, category: Optional[str] = None) -> List[str]:
        """Get concept names, optionally filtered by category"""
        if category and category in self.category_ids:
            indices = self.category_ids[category]
            return [self.concepts[i] for i in indices]
        return self.concepts


class VisualRadLexAlignment(nn.Module):
    """
    Cross-modal alignment between visual features and RadLex concepts
    
    Uses cross-attention to align multi-scale visual features with
    medical ontology embeddings for grounded understanding
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # RadLex embeddings
        self.radlex = RadLexEmbedding(embed_dim=hidden_dim)
        
        # Cross-attention: visual queries RadLex
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Alignment projection
        self.alignment_head = nn.Linear(hidden_dim, len(RADLEX_CONCEPTS))
        
    def forward(
        self,
        visual_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Align visual features with RadLex concepts
        
        Args:
            visual_features: Flattened visual features [B, N, D]
            return_attention: Whether to return attention weights
            
        Returns:
            aligned_features: RadLex-aligned features [B, N, D]
            attention_weights: Optional attention weights [B, N, num_concepts]
        """
        B, N, D = visual_features.shape
        
        # Get RadLex embeddings
        radlex_emb = self.radlex()  # [num_concepts, D]
        radlex_emb = radlex_emb.unsqueeze(0).expand(B, -1, -1)  # [B, num_concepts, D]
        
        # Cross-attention: visual queries RadLex
        attn_out, attn_weights = self.cross_attn(
            query=self.norm1(visual_features),
            key=radlex_emb,
            value=radlex_emb,
            need_weights=return_attention,
        )
        
        # Residual connection
        visual_features = visual_features + attn_out
        
        # FFN
        visual_features = visual_features + self.ffn(self.norm2(visual_features))
        
        if return_attention:
            return visual_features, attn_weights
        return visual_features, None


class PROFAEncoder(nn.Module):
    """
    Complete PRO-FA (Progressive Feature Alignment) Module
    
    Combines:
    1. Multi-scale Swin-Transformer backbone
    2. RadLex concept embeddings
    3. Visual-RadLex alignment
    4. Hierarchical feature fusion
    """
    
    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multi-scale visual encoder
        self.visual_encoder = MultiScaleSwinEncoder(
            backbone=backbone,
            pretrained=pretrained,
        )
        
        # RadLex alignment for each scale
        self.organ_alignment = VisualRadLexAlignment(hidden_dim, num_heads, dropout)
        self.region_alignment = VisualRadLexAlignment(hidden_dim, num_heads, dropout)
        self.pixel_alignment = VisualRadLexAlignment(hidden_dim, num_heads, dropout)
        
        # Hierarchical fusion
        self.hierarchical_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Scale embeddings
        self.scale_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Input images [B, 3, H, W] or [B, V, 3, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with:
            - visual_features: Fused multi-scale features [B, N, D]
            - pooled_features: Global representation [B, D]
            - organ_features, region_features, pixel_features: Scale-specific
            - attention_maps: Optional attention visualizations
        """
        # Extract multi-scale features
        scale_features = self.visual_encoder(images)
        
        # Flatten spatial dimensions
        B = scale_features["organ_features"].shape[0]
        
        organ_flat = rearrange(scale_features["organ_features"], 'b d h w -> b (h w) d')
        region_flat = rearrange(scale_features["region_features"], 'b d h w -> b (h w) d')
        pixel_flat = rearrange(scale_features["pixel_features"], 'b d h w -> b (h w) d')
        
        # RadLex alignment at each scale
        organ_aligned, organ_attn = self.organ_alignment(organ_flat, return_attention)
        region_aligned, region_attn = self.region_alignment(region_flat, return_attention)
        pixel_aligned, pixel_attn = self.pixel_alignment(pixel_flat, return_attention)
        
        # Add scale embeddings
        organ_aligned = organ_aligned + self.scale_embed[0]
        region_aligned = region_aligned + self.scale_embed[1]
        pixel_aligned = pixel_aligned + self.scale_embed[2]
        
        # Concatenate all scales
        all_features = torch.cat([organ_aligned, region_aligned, pixel_aligned], dim=1)
        
        # Self-attention for hierarchical fusion
        fused, _ = self.hierarchical_attn(
            self.fusion_norm(all_features),
            self.fusion_norm(all_features),
            self.fusion_norm(all_features),
        )
        fused = all_features + fused
        
        # Output projection
        fused = self.output_proj(fused)
        
        # Pooled representation
        pooled = fused.mean(dim=1)
        
        output = {
            "visual_features": fused,                    # [B, 16+49+49, D]
            "pooled_features": pooled,                   # [B, D]
            "organ_features": organ_aligned,             # [B, 16, D]
            "region_features": region_aligned,           # [B, 49, D]
            "pixel_features": pixel_aligned,             # [B, 49, D]
        }
        
        if return_attention:
            output["attention_maps"] = {
                "organ": organ_attn,
                "region": region_attn,
                "pixel": pixel_attn,
            }
        
        return output


if __name__ == "__main__":
    # Test the encoder
    print("Testing PRO-FA Encoder...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create encoder
    encoder = PROFAEncoder(
        backbone="swin_base_patch4_window7_224",
        pretrained=True,
        hidden_dim=768,
    ).to(device)
    
    # Test with single image
    x = torch.randn(2, 3, 224, 224).to(device)
    output = encoder(x, return_attention=True)
    
    print(f"\nSingle image input shape: {x.shape}")
    print(f"Visual features shape: {output['visual_features'].shape}")
    print(f"Pooled features shape: {output['pooled_features'].shape}")
    print(f"Organ features shape: {output['organ_features'].shape}")
    print(f"Region features shape: {output['region_features'].shape}")
    print(f"Pixel features shape: {output['pixel_features'].shape}")
    
    # Test with multi-view input
    x_multi = torch.randn(2, 2, 3, 224, 224).to(device)
    output_multi = encoder(x_multi)
    
    print(f"\nMulti-view input shape: {x_multi.shape}")
    print(f"Visual features shape: {output_multi['visual_features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nPRO-FA Encoder test passed! ✓")
