"""
RCTA: Triangular Cognitive Attention and Report Generator Module

Implements closed-loop cognitive verification for radiology report generation:
1. Image → Clinical Text: Context creation
2. Context → Predicted Labels: Hypothesis formation
3. Hypothesis → Image: Verification loop

Plus GPT-2 based medical text decoder for Findings/Impression generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from einops import rearrange, repeat
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer for querying between modalities
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor [B, N_q, D]
            key_value: Key-value tensor [B, N_kv, D]
            key_padding_mask: Mask for key positions [B, N_kv]
            
        Returns:
            output: Attended output [B, N_q, D]
            attention: Optional attention weights [B, H, N_q, N_kv]
        """
        B, N_q, D = query.shape
        _, N_kv, _ = key_value.shape
        
        # Layer norm
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, N_q, N_kv]
        
        # Apply mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class TriangularCognitiveAttention(nn.Module):
    """
    RCTA: Triangular Cognitive Attention Module
    
    Implements cognitive verification loop:
    1. Visual features query Clinical text → Context
    2. Context queries Predicted labels → Hypothesis
    3. Hypothesis queries Visual features → Verified representation
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stage 1: Image → Clinical Text (Context Creation)
        self.image_to_text_attn = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Stage 2: Context → Labels (Hypothesis Formation)
        self.context_to_labels_attn = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Stage 3: Hypothesis → Image (Verification)
        self.hypothesis_to_image_attn = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # FFN blocks for each stage
        self.ffn_context = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        
        self.ffn_hypothesis = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        
        self.ffn_verified = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        
        # Clinical text encoder (for indication/history)
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Label embedding projection
        self.label_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        label_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: PRO-FA output [B, N_v, D]
            text_features: Clinical indication embeddings [B, N_t, D]
            label_embeddings: MIX-MLP label embeddings [B, 14, D]
            text_mask: Padding mask for text [B, N_t]
            
        Returns:
            Dictionary with:
            - verified_features: Final verified representation [B, N, D]
            - context: Intermediate context [B, N, D]
            - hypothesis: Intermediate hypothesis [B, N, D]
        """
        B = visual_features.shape[0]
        
        # Project text and labels
        text_features = self.text_proj(text_features)
        label_embeddings = self.label_proj(label_embeddings)
        
        intermediates = {"contexts": [], "hypotheses": [], "verified": []}
        
        # Multi-layer triangular attention
        context = visual_features
        for i in range(self.num_layers):
            # Stage 1: Image → Clinical Text (Context)
            attn_out, _ = self.image_to_text_attn[i](
                context, text_features, text_mask
            )
            context = context + attn_out
            context = context + self.ffn_context[i](context)
            
            if return_intermediates:
                intermediates["contexts"].append(context)
            
            # Stage 2: Context → Labels (Hypothesis)
            attn_out, _ = self.context_to_labels_attn[i](
                context, label_embeddings
            )
            hypothesis = context + attn_out
            hypothesis = hypothesis + self.ffn_hypothesis[i](hypothesis)
            
            if return_intermediates:
                intermediates["hypotheses"].append(hypothesis)
            
            # Stage 3: Hypothesis → Image (Verification)
            attn_out, _ = self.hypothesis_to_image_attn[i](
                hypothesis, visual_features
            )
            verified = hypothesis + attn_out
            verified = verified + self.ffn_verified[i](verified)
            
            if return_intermediates:
                intermediates["verified"].append(verified)
            
            # Use verified as input for next layer's context
            context = verified
        
        # Final output
        output = self.output_norm(self.output_proj(verified))
        
        result = {
            "verified_features": output,
            "context": context,
            "hypothesis": hypothesis,
        }
        
        if return_intermediates:
            result["intermediates"] = intermediates
        
        return result


class MedicalReportGenerator(nn.Module):
    """
    GPT-2 based medical report generator
    
    Features:
    - Conditioned on RCTA-verified visual features
    - Separate generation for Findings and Impression
    - Medical vocabulary focus
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 50257,
        max_length: int = 256,
        num_layers: int = 12,
        num_heads: int = 12,
        pretrained_model: str = "gpt2",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model)
        gpt2_hidden = self.gpt2.config.hidden_size
        
        # Visual feature projection to GPT-2 dimension
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, gpt2_hidden),
            nn.LayerNorm(gpt2_hidden),
        )
        
        # Learnable prefix tokens for conditioning
        self.num_prefix = 8
        self.prefix_embed = nn.Parameter(
            torch.randn(self.num_prefix, gpt2_hidden) * 0.02
        )
        
        # Section embeddings (Findings vs Impression)
        self.section_embed = nn.Embedding(2, gpt2_hidden)
        
        # Cross-attention adapter for visual conditioning
        self.cross_attn = nn.ModuleList([
            CrossModalAttention(gpt2_hidden, 12, dropout)
            for _ in range(4)  # Add cross-attention every 3 layers
        ])
        
        # Integration layers
        self.cross_attn_layers = [2, 5, 8, 11]  # Which GPT-2 layers to add cross-attn
        
        # Output heads
        self.findings_head = nn.Linear(gpt2_hidden, vocab_size, bias=False)
        self.impression_head = nn.Linear(gpt2_hidden, vocab_size, bias=False)
        
        # Share weights with GPT-2 embeddings
        self.findings_head.weight = self.gpt2.lm_head.weight
        self.impression_head.weight = self.gpt2.lm_head.weight
        
    def forward(
        self,
        verified_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        section: int = 0,  # 0 = findings, 1 = impression
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            verified_features: RCTA output [B, N, D]
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            labels: Target IDs for loss [B, L]
            section: 0 for findings, 1 for impression
            
        Returns:
            Dictionary with logits and optional loss
        """
        B, N, _ = verified_features.shape
        L = input_ids.shape[1]
        
        # Project visual features
        visual_cond = self.visual_proj(verified_features)  # [B, N, D_gpt2]
        
        # Add prefix tokens
        prefix = self.prefix_embed.unsqueeze(0).expand(B, -1, -1)  # [B, num_prefix, D]
        visual_cond = torch.cat([prefix, visual_cond], dim=1)  # [B, num_prefix + N, D]
        
        # Get text embeddings
        text_embeds = self.gpt2.transformer.wte(input_ids)  # [B, L, D]
        
        # Add section embedding
        section_emb = self.section_embed(
            torch.full((B,), section, device=input_ids.device, dtype=torch.long)
        )
        text_embeds = text_embeds + section_emb.unsqueeze(1)
        
        # Concatenate visual conditioning with text
        inputs_embeds = torch.cat([visual_cond, text_embeds], dim=1)
        
        # Create attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(B, visual_cond.size(1), device=attention_mask.device)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            full_mask = None
        
        # Forward through GPT-2
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
        )
        
        # Get logits for text portion only
        hidden_states = outputs.hidden_states[-1]
        text_hidden = hidden_states[:, visual_cond.size(1):, :]  # [B, L, D]
        
        # Apply section-specific head
        if section == 0:
            logits = self.findings_head(text_hidden)
        else:
            logits = self.impression_head(text_hidden)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        verified_features: torch.Tensor,
        tokenizer,
        max_length: int = 128,
        section: int = 0,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        """
        Generate radiology report text
        
        Args:
            verified_features: RCTA output [B, N, D]
            tokenizer: Tokenizer for decoding
            max_length: Maximum generation length
            section: 0 for findings, 1 for impression
            
        Returns:
            List of generated strings
        """
        B = verified_features.shape[0]
        device = verified_features.device
        
        # Project visual features
        visual_cond = self.visual_proj(verified_features)
        
        # Add prefix
        prefix = self.prefix_embed.unsqueeze(0).expand(B, -1, -1)
        visual_cond = torch.cat([prefix, visual_cond], dim=1)
        
        # Section prompt
        if section == 0:
            prompt = "FINDINGS:"
        else:
            prompt = "IMPRESSION:"
        
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_ids = prompt_ids.expand(B, -1)
        
        # Get prompt embeddings
        prompt_embeds = self.gpt2.transformer.wte(prompt_ids)
        section_emb = self.section_embed(
            torch.full((B,), section, device=device, dtype=torch.long)
        )
        prompt_embeds = prompt_embeds + section_emb.unsqueeze(1)
        
        # Concatenate with visual conditioning
        inputs_embeds = torch.cat([visual_cond, prompt_embeds], dim=1)
        
        # Generate
        generated = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length + inputs_embeds.size(1),
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
        )
        
        # Decode (skip visual prefix tokens)
        generated_text = []
        for i in range(B):
            # The generated sequence includes the visual prefix
            text = tokenizer.decode(
                generated[i][visual_cond.size(1):],
                skip_special_tokens=True
            )
            generated_text.append(text)
        
        return generated_text


class RCTADecoder(nn.Module):
    """
    Complete RCTA + Report Generator Module
    
    Combines:
    1. Triangular Cognitive Attention for verification
    2. Medical Report Generator for text output
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_rcta_layers: int = 3,
        pretrained_generator: str = "gpt2",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # RCTA module
        self.rcta = TriangularCognitiveAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_rcta_layers,
            dropout=dropout,
        )
        
        # Report generator
        self.generator = MedicalReportGenerator(
            hidden_dim=hidden_dim,
            pretrained_model=pretrained_generator,
            dropout=dropout,
        )
        
        # Clinical text encoder
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # Token embedding for indication
        self.token_embed = nn.Embedding(50257, hidden_dim)  # GPT-2 vocab size
        
    def forward(
        self,
        visual_features: torch.Tensor,
        label_embeddings: torch.Tensor,
        indication_ids: torch.Tensor,
        indication_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            visual_features: PRO-FA output [B, N_v, D]
            label_embeddings: MIX-MLP label embeddings [B, 14, D]
            indication_ids: Clinical indication token IDs [B, N_t]
            indication_mask: Mask for indication [B, N_t]
            input_ids: Target report tokens [B, L]
            attention_mask: Target mask [B, L]
            labels: Target labels for loss [B, L]
            
        Returns:
            Dictionary with verified features, logits, and loss
        """
        # Encode clinical indication
        text_embeds = self.token_embed(indication_ids)
        text_features = self.text_encoder(
            text_embeds,
            src_key_padding_mask=(indication_mask == 0),
        )
        
        # RCTA verification
        rcta_output = self.rcta(
            visual_features=visual_features,
            text_features=text_features,
            label_embeddings=label_embeddings,
            text_mask=(indication_mask == 0),
        )
        
        verified_features = rcta_output["verified_features"]
        
        # Generate report
        gen_output = self.generator(
            verified_features=verified_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "verified_features": verified_features,
            "context": rcta_output["context"],
            "hypothesis": rcta_output["hypothesis"],
            "logits": gen_output["logits"],
            "loss": gen_output.get("loss"),
        }
    
    @torch.no_grad()
    def generate_report(
        self,
        visual_features: torch.Tensor,
        label_embeddings: torch.Tensor,
        indication_ids: torch.Tensor,
        indication_mask: torch.Tensor,
        tokenizer,
        max_findings_length: int = 150,
        max_impression_length: int = 75,
    ) -> Dict[str, List[str]]:
        """
        Generate complete radiology report
        
        Returns:
            Dictionary with findings and impression texts
        """
        # Encode indication
        text_embeds = self.token_embed(indication_ids)
        text_features = self.text_encoder(
            text_embeds,
            src_key_padding_mask=(indication_mask == 0),
        )
        
        # RCTA verification
        rcta_output = self.rcta(
            visual_features=visual_features,
            text_features=text_features,
            label_embeddings=label_embeddings,
            text_mask=(indication_mask == 0),
        )
        
        verified_features = rcta_output["verified_features"]
        
        # Generate findings
        findings = self.generator.generate(
            verified_features=verified_features,
            tokenizer=tokenizer,
            max_length=max_findings_length,
            section=0,
        )
        
        # Generate impression
        impressions = self.generator.generate(
            verified_features=verified_features,
            tokenizer=tokenizer,
            max_length=max_impression_length,
            section=1,
        )
        
        return {
            "findings": findings,
            "impressions": impressions,
        }


if __name__ == "__main__":
    # Test the decoder
    print("Testing RCTA Decoder...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create decoder
    decoder = RCTADecoder(
        hidden_dim=768,
        num_heads=12,
        num_rcta_layers=3,
        pretrained_generator="gpt2",
    ).to(device)
    
    # Test inputs
    B = 2
    visual_features = torch.randn(B, 114, 768).to(device)  # 16 + 49 + 49
    label_embeddings = torch.randn(B, 14, 768).to(device)
    indication_ids = torch.randint(0, 1000, (B, 32)).to(device)
    indication_mask = torch.ones(B, 32).to(device)
    input_ids = torch.randint(0, 1000, (B, 64)).to(device)
    attention_mask = torch.ones(B, 64).to(device)
    labels = input_ids.clone()
    
    # Forward pass
    output = decoder(
        visual_features=visual_features,
        label_embeddings=label_embeddings,
        indication_ids=indication_ids,
        indication_mask=indication_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    print(f"\nVisual features shape: {visual_features.shape}")
    print(f"Verified features shape: {output['verified_features'].shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nRCTA Decoder test passed! ✓")
