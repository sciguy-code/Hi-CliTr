"""
PyTorch Dataset Classes for Radiology Report Generation
Supports IU-Xray and MIMIC-CXR datasets
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, RADLEX_CONCEPTS


class IUXrayDataset(Dataset):
    """
    IU-Xray Dataset for Radiology Report Generation
    
    Features:
    - Multi-view support (Frontal + Lateral)
    - RadLex concept extraction
    - CheXpert label extraction (rule-based)
    - Configurable text targets (findings, impression, or full report)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        max_seq_length: int = 256,
        tokenizer_name: str = "gpt2",
        transform: Optional[transforms.Compose] = None,
        return_multi_view: bool = True,
        target_type: str = "report",  # "findings", "impression", or "report"
    ):
        """
        Args:
            data_dir: Path to IU-Xray dataset
            split: One of "train", "val", "test"
            image_size: Image resize dimension
            max_seq_length: Maximum sequence length for tokenization
            tokenizer_name: HuggingFace tokenizer name
            transform: Optional custom transforms
            return_multi_view: Whether to return multiple views per study
            target_type: Which text to use as target
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.return_multi_view = return_multi_view
        self.target_type = target_type
        
        # Load processed data
        self.data = self._load_data()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Image transforms
        self.transform = transform or self._default_transforms(split == "train")
        
        # Image directory
        self.image_dir = self._find_image_dir()
        
        # Group by study (uid) for multi-view
        if return_multi_view:
            self.study_groups = self.data.groupby("uid")
            self.study_ids = list(self.study_groups.groups.keys())
        
        # CheXpert labels from indication/findings
        self.chexpert_labels = self._extract_chexpert_labels()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and filter data for the specified split"""
        processed_file = self.data_dir / "processed_data.csv"
        
        if processed_file.exists():
            df = pd.read_csv(processed_file)
        else:
            # Fall back to raw files
            projections = pd.read_csv(self.data_dir / "indiana_projections.csv")
            reports = pd.read_csv(self.data_dir / "indiana_reports.csv")
            df = projections.merge(reports, on="uid", how="inner")
            
            # Create split manually
            unique_uids = df["uid"].unique()
            np.random.seed(42)
            np.random.shuffle(unique_uids)
            
            n_total = len(unique_uids)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            
            train_uids = set(unique_uids[:n_train])
            val_uids = set(unique_uids[n_train:n_train + n_val])
            
            df["split"] = df["uid"].apply(
                lambda x: "train" if x in train_uids else ("val" if x in val_uids else "test")
            )
        
        # Filter by split
        df = df[df["split"] == self.split].reset_index(drop=True)
        
        # Fill NaN values
        text_cols = ["findings", "impression", "indication"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("")
        
        return df
    
    def _find_image_dir(self) -> Path:
        """Find the image directory"""
        candidates = ["images", "images_normalized", "Images"]
        for candidate in candidates:
            path = self.data_dir / candidate
            if path.exists():
                return path
        raise FileNotFoundError(f"No image directory found in {self.data_dir}")
    
    def _default_transforms(self, is_train: bool) -> transforms.Compose:
        """Default image transforms"""
        config = get_config()
        
        if is_train:
            return transforms.Compose([
                transforms.Resize((self.image_size + 32, self.image_size + 32)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.data.normalize_mean,
                    std=config.data.normalize_std
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.data.normalize_mean,
                    std=config.data.normalize_std
                )
            ])
    
    def _extract_chexpert_labels(self) -> Dict[str, torch.Tensor]:
        """
        Extract CheXpert-style labels from text using rule-based matching
        """
        config = get_config()
        labels = config.classifier.chexpert_labels
        
        # Simple keyword matching (production should use chexpert-labeler)
        keywords = {
            "No Finding": ["normal", "no acute", "unremarkable", "clear"],
            "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
            "Lung Opacity": ["opacity", "opacit", "infiltrate", "haziness"],
            "Pleural Effusion": ["effusion", "pleural fluid"],
            "Edema": ["edema", "congestion", "pulmonary edema"],
            "Consolidation": ["consolidation", "consolidative"],
            "Atelectasis": ["atelectasis", "atelectatic", "volume loss"],
            "Pneumonia": ["pneumonia", "infectious"],
            "Pneumothorax": ["pneumothorax"],
            "Fracture": ["fracture", "fractured"],
            "Lung Lesion": ["lesion", "mass", "nodule", "tumor"],
            "Pleural Other": ["pleural thickening", "pleural calcification"],
            "Enlarged Cardiomediastinum": ["mediastinal widening", "mediastinum"],
            "Support Devices": ["tube", "catheter", "pacemaker", "device", "wire"]
        }
        
        study_labels = {}
        
        for _, row in self.data.iterrows():
            uid = row["uid"]
            text = (str(row.get("findings", "")) + " " + 
                    str(row.get("impression", ""))).lower()
            
            label_vec = torch.zeros(len(labels))
            
            for i, label in enumerate(labels):
                if label in keywords:
                    for keyword in keywords[label]:
                        if keyword in text:
                            label_vec[i] = 1.0
                            break
            
            study_labels[uid] = label_vec
        
        return study_labels
    
    def _load_image(self, filename: str) -> torch.Tensor:
        """Load and transform a single image"""
        # Handle various filename formats
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename = filename + '.png'
        
        image_path = self.image_dir / filename
        
        # Try with different extensions
        if not image_path.exists():
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_path = self.image_dir / (filename.rsplit('.', 1)[0] + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            # Return black image if not found
            return torch.zeros(3, self.image_size, self.image_size)
        
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _get_text_target(self, row: pd.Series) -> str:
        """Get target text based on target_type"""
        if self.target_type == "findings":
            return str(row.get("findings", ""))
        elif self.target_type == "impression":
            return str(row.get("impression", ""))
        else:  # "report"
            findings = str(row.get("findings", ""))
            impression = str(row.get("impression", ""))
            parts = []
            if findings:
                parts.append(f"FINDINGS: {findings}")
            if impression:
                parts.append(f"IMPRESSION: {impression}")
            return " ".join(parts)
    
    def __len__(self) -> int:
        if self.return_multi_view:
            return len(self.study_ids)
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.return_multi_view:
            # Get study and all its images
            uid = self.study_ids[idx]
            study_data = self.study_groups.get_group(uid)
            
            # Load all images for this study
            images = []
            for _, row in study_data.iterrows():
                filename = row["filename"]
                images.append(self._load_image(filename))
            
            # Pad/truncate to 2 views (frontal + lateral)
            if len(images) < 2:
                images.append(torch.zeros_like(images[0]))
            images = images[:2]
            
            # Stack images
            images = torch.stack(images)  # [2, 3, H, W]
            
            # Get text from first row
            first_row = study_data.iloc[0]
        else:
            row = self.data.iloc[idx]
            uid = row["uid"]
            images = self._load_image(row["filename"]).unsqueeze(0)  # [1, 3, H, W]
            first_row = row
        
        # Get target text
        target_text = self._get_text_target(first_row)
        
        # Tokenize
        encoding = self.tokenizer(
            target_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get clinical indication
        indication = str(first_row.get("indication", ""))
        indication_encoding = self.tokenizer(
            indication if indication else "No clinical indication provided.",
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get CheXpert labels
        labels = self.chexpert_labels.get(uid, torch.zeros(14))
        
        return {
            "uid": uid,
            "images": images,  # [num_views, 3, H, W]
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "indication_ids": indication_encoding["input_ids"].squeeze(0),
            "indication_mask": indication_encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "target_text": target_text,
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    max_seq_length: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = IUXrayDataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        max_seq_length=max_seq_length,
    )
    
    val_dataset = IUXrayDataset(
        data_dir=data_dir,
        split="val",
        image_size=image_size,
        max_seq_length=max_seq_length,
    )
    
    test_dataset = IUXrayDataset(
        data_dir=data_dir,
        split="test",
        image_size=image_size,
        max_seq_length=max_seq_length,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing IUXrayDataset...")
    
    dataset = IUXrayDataset(
        data_dir="./data/iu_xray",
        split="train",
        return_multi_view=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Target text preview: {sample['target_text'][:100]}...")
