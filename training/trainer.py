"""
Training Pipeline for Cognitive Radiology Model

Implements:
- Multi-task training (classification + generation)
- Mixed precision training
- Learning rate scheduling with warmup
- Gradient accumulation
- Checkpointing and early stopping
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path
from tqdm import tqdm
import wandb

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models.model import CognitiveRadiologyModel, create_model
from models.classifier import UncertaintyAwareLoss
from data.dataset import create_dataloaders


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr_mult = self._get_lr_multiplier()
        
        for idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_lrs[idx] * lr_mult
    
    def _get_lr_multiplier(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + torch.cos(torch.tensor(progress * 3.14159)).item()
            )
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    """
    Trainer for Cognitive Radiology Model
    
    Features:
    - Multi-task loss (classification + generation)
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Checkpointing
    """
    
    def __init__(
        self,
        model: CognitiveRadiologyModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config=None,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or get_config()
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        steps_per_epoch = len(train_loader) // self.config.training.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.training.max_epochs
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.config.training.min_lr_ratio,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.training.use_amp else None
        
        # Loss functions
        self.cls_loss_fn = UncertaintyAwareLoss(strategy="zeros")
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Logging
        if self.use_wandb:
            wandb.init(
                project="cognitive-radiology",
                config=vars(self.config),
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with layer-wise learning rate decay"""
        # Separate parameters for different components
        encoder_params = list(self.model.encoder.parameters())
        classifier_params = list(self.model.classifier.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        # Lower LR for pretrained encoder
        param_groups = [
            {
                "params": encoder_params,
                "lr": self.config.training.learning_rate * 0.1,  # Lower for pretrained
            },
            {
                "params": classifier_params,
                "lr": self.config.training.learning_rate,
            },
            {
                "params": decoder_params,
                "lr": self.config.training.learning_rate,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
        )
        
        return optimizer
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0
        
        accumulation_steps = self.config.training.gradient_accumulation_steps
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.max_epochs}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["images"].to(self.device)
            indication_ids = batch["indication_ids"].to(self.device)
            indication_mask = batch["indication_mask"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.config.training.use_amp):
                output = self.model(
                    images=images,
                    indication_ids=indication_ids,
                    indication_mask=indication_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,  # Use input_ids as labels for LM
                    chexpert_labels=labels,
                )
                
                loss = output["total_loss"] / accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip_norm,
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += output["total_loss"].item()
            if "loss_cls" in output:
                total_cls_loss += output["loss_cls"].item()
            if "loss_gen" in output:
                total_gen_loss += output["loss_gen"].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{self.scheduler.get_lr():.2e}",
            })
        
        metrics = {
            "train/loss": total_loss / num_batches,
            "train/cls_loss": total_cls_loss / num_batches,
            "train/gen_loss": total_gen_loss / num_batches,
            "train/lr": self.scheduler.get_lr(),
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            # Move to device
            images = batch["images"].to(self.device)
            indication_ids = batch["indication_ids"].to(self.device)
            indication_mask = batch["indication_mask"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.config.training.use_amp):
                output = self.model(
                    images=images,
                    indication_ids=indication_ids,
                    indication_mask=indication_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    chexpert_labels=labels,
                )
            
            # Accumulate metrics
            total_loss += output["total_loss"].item()
            if "loss_cls" in output:
                total_cls_loss += output["loss_cls"].item()
            if "loss_gen" in output:
                total_gen_loss += output["loss_gen"].item()
            num_batches += 1
            
            # Collect predictions for F1
            preds = (output["classification_probs"] > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        
        # Calculate F1
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Micro F1
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics = {
            "val/loss": total_loss / num_batches,
            "val/cls_loss": total_cls_loss / num_batches,
            "val/gen_loss": total_gen_loss / num_batches,
            "val/chexpert_f1": f1.item(),
            "val/precision": precision.item(),
            "val/recall": recall.item(),
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.current_step,
            "metrics": metrics,
            "config": vars(self.config),
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
        
        # Save periodic
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch + 1}.pt")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.current_step = checkpoint["scheduler_step"]
        
        return checkpoint["epoch"], checkpoint["metrics"]
    
    def train(self) -> Dict[str, float]:
        """Full training loop"""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training.max_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.training.batch_size * self.config.training.gradient_accumulation_steps}")
        print("=" * 60)
        
        best_metrics = {}
        
        for epoch in range(self.config.training.max_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{self.config.training.max_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train/loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val/loss']:.4f}")
            print(f"  Val F1: {val_metrics['val/chexpert_f1']:.4f}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Check for best model
            is_best = val_metrics["val/loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val/loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                best_metrics = val_metrics.copy()
                print(f"  ** New best model! **")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best epoch: {self.best_epoch + 1}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        if self.use_wandb:
            wandb.finish()
        
        return best_metrics


def train_model(
    data_dir: str = "./data/iu_xray",
    checkpoint_dir: str = "./checkpoints",
    max_epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    use_wandb: bool = False,
    fast_dev_run: bool = False,
):
    """
    Main training function
    
    Args:
        data_dir: Path to dataset
        checkpoint_dir: Path for saving checkpoints
        max_epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        use_wandb: Whether to use Weights & Biases
        fast_dev_run: Quick test run
    """
    # Load config
    config = get_config()
    config.training.max_epochs = max_epochs if not fast_dev_run else 1
    config.training.batch_size = batch_size
    config.training.learning_rate = learning_rate
    config.training.checkpoint_dir = checkpoint_dir
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4 if not fast_dev_run else 0,
        image_size=config.data.image_size,
        max_seq_length=config.data.max_seq_length,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config, pretrained=True, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=use_wandb,
    )
    
    # Train
    best_metrics = trainer.train()
    
    return best_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Cognitive Radiology Model")
    parser.add_argument("--data_dir", type=str, default="./data/iu_xray")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        use_wandb=args.use_wandb,
        fast_dev_run=args.fast_dev_run,
    )
