"""
PyTorch Dataset Classes for Radiology Report Generation
Supports IU-Xray dataset with Study object structure
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
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, RADLEX_CONCEPTS, IU_DIR, IU_REPORTS_CSV, IU_PROJECTIONS_CSV, IU_IMAGES_DIR


@dataclass
class Study:
    """
    Unified Study object structure for IU-Xray dataset
    
    Structure:
    {
        "images": {
            "frontal": str | None,  # Path to frontal image
            "lateral": str | None   # Path to lateral image
        },
        "indication": str,
        "findings": str,
        "impression": str,
        "labels": None,  # Can be populated with CheXpert labels later
        "source": "iu"
    }
    """
    images: Dict[str, Optional[str]]
    indication: str
    findings: str
    impression: str
    labels: Optional[torch.Tensor] = None
    source: str = "iu"
    
    def to_dict(self) -> Dict:
        """Convert Study to dictionary"""
        return {
            "images": self.images,
            "indication": self.indication,
            "findings": self.findings,
            "impression": self.impression,
            "labels": self.labels,
            "source": self.source
        }


def load_iu_xray_studies(data_dir: Optional[str] = None, auto_extract: bool = True) -> List[Study]:
    """
    Load IU-Xray dataset and convert to Study objects.
    
    This function:
    - Automatically extracts from zip if dataset not found
    - Reads indiana_reports.csv
    - Reads indiana_projections.csv
    - Merges them on uid
    - Groups multiple images per study (frontal / lateral)
    - Converts every study into a unified Study object
    
    Args:
        data_dir: Optional override for dataset directory (defaults to IU_DIR from config)
        auto_extract: If True, automatically extract zip files if dataset not found
    
    Returns:
        List of Study objects
    
    Raises:
        FileNotFoundError: If required CSV files are missing
        ValueError: If data cannot be merged or processed
    """
    if data_dir is None:
        data_dir = Path(IU_DIR)
    else:
        data_dir = Path(data_dir)
    
    # Ensure dataset is extracted first (import here to avoid circular import)
    if auto_extract:
        try:
            from data.download_iu_xray import ensure_dataset_extracted
            data_dir = ensure_dataset_extracted(str(data_dir), auto_extract=True)
        except ImportError:
            # If import fails, just proceed (extraction will be handled by verify)
            pass
    
    # Fail fast: Check required files
    reports_csv = data_dir / "indiana_reports.csv"
    projections_csv = data_dir / "indiana_projections.csv"
    images_dir = data_dir / "images"
    
    if not reports_csv.exists():
        raise FileNotFoundError(
            f"Required file not found: {reports_csv}\n"
            f"Expected location: {reports_csv.absolute()}\n"
            f"Please ensure indiana_reports.csv exists in the dataset directory."
        )
    
    if not projections_csv.exists():
        raise FileNotFoundError(
            f"Required file not found: {projections_csv}\n"
            f"Expected location: {projections_csv.absolute()}\n"
            f"Please ensure indiana_projections.csv exists in the dataset directory."
        )
    
    # Try to find images directory (case-insensitive, including nested structures)
    # First check nested structures (e.g., images/images_normalized/)
    found_images_dir = None
    for base_name in ["images", "Images"]:
        base_path = data_dir / base_name
        if base_path.exists() and base_path.is_dir():
            # Check nested directories first
            for nested_name in ["images_normalized", "Images", "images"]:
                nested_path = base_path / nested_name
                if nested_path.exists() and nested_path.is_dir():
                    # Check if it has images
                    num_images = len(list(nested_path.glob("*.png"))) + len(list(nested_path.glob("*.jpg"))) + len(list(nested_path.glob("*.jpeg")))
                    if num_images > 0:
                        found_images_dir = nested_path
                        break
            # Also check if base_path itself has images
            if found_images_dir is None:
                num_images = len(list(base_path.glob("*.png"))) + len(list(base_path.glob("*.jpg"))) + len(list(base_path.glob("*.jpeg")))
                if num_images > 0:
                    found_images_dir = base_path
    
    # Try direct directories
    if found_images_dir is None:
        for alt_name in ["Images", "images_normalized"]:
            alt_path = data_dir / alt_name
            if alt_path.exists() and alt_path.is_dir():
                num_images = len(list(alt_path.glob("*.png"))) + len(list(alt_path.glob("*.jpg"))) + len(list(alt_path.glob("*.jpeg")))
                if num_images > 0:
                    found_images_dir = alt_path
                    break
    
    if found_images_dir is not None:
        images_dir = found_images_dir
    else:
        raise FileNotFoundError(
            f"Image directory not found in {data_dir.absolute()}\n"
            f"Expected one of: images/, Images/, images_normalized/, or images/images_normalized/\n"
            f"Please ensure images are extracted to the dataset directory."
        )
    
    # Load CSV files
    try:
        reports_df = pd.read_csv(reports_csv)
        projections_df = pd.read_csv(projections_csv)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV files: {e}")
    
    # Validate required columns
    required_report_cols = ["uid"]
    required_projection_cols = ["uid", "filename"]
    
    missing_report_cols = [col for col in required_report_cols if col not in reports_df.columns]
    if missing_report_cols:
        raise ValueError(
            f"indiana_reports.csv missing required columns: {missing_report_cols}\n"
            f"Available columns: {list(reports_df.columns)}"
        )
    
    missing_projection_cols = [col for col in required_projection_cols if col not in projections_df.columns]
    if missing_projection_cols:
        raise ValueError(
            f"indiana_projections.csv missing required columns: {missing_projection_cols}\n"
            f"Available columns: {list(projections_df.columns)}"
        )
    
    # Merge on uid
    merged_df = projections_df.merge(reports_df, on="uid", how="inner")
    
    if len(merged_df) == 0:
        raise ValueError(
            "No matching UIDs found between projections and reports.\n"
            "Check that both CSV files contain valid 'uid' columns."
        )
    
    # Fill NaN values in text columns
    text_cols = ["findings", "impression", "indication"]
    for col in text_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("").astype(str)
        else:
            merged_df[col] = ""
    
    # Group by study (uid) to handle multiple images per study
    studies = []
    
    for uid, group in merged_df.groupby("uid"):
        # Get text fields from first row (should be same for all rows of same uid)
        first_row = group.iloc[0]
        indication = str(first_row.get("indication", "")).strip()
        findings = str(first_row.get("findings", "")).strip()
        impression = str(first_row.get("impression", "")).strip()
        
        # Group images by view type (frontal/lateral)
        frontal_path = None
        lateral_path = None
        
        # Check for view type column
        if "view" in group.columns or "projection" in group.columns:
            view_col = "view" if "view" in group.columns else "projection"
            for _, row in group.iterrows():
                view_type = str(row.get(view_col, "")).lower()
                filename = str(row["filename"])
                
                # Determine if frontal or lateral
                if "frontal" in view_type or "pa" in view_type or "ap" in view_type:
                    if frontal_path is None:
                        frontal_path = filename
                elif "lateral" in view_type or "lat" in view_type:
                    if lateral_path is None:
                        lateral_path = filename
                else:
                    # If no view type specified, assign first to frontal, second to lateral
                    if frontal_path is None:
                        frontal_path = filename
                    elif lateral_path is None:
                        lateral_path = filename
        else:
            # No view type column - assign first image to frontal, second to lateral
            filenames = group["filename"].tolist()
            if len(filenames) > 0:
                frontal_path = filenames[0]
            if len(filenames) > 1:
                lateral_path = filenames[1]
        
        # Verify image files exist (fail fast)
        if frontal_path:
            frontal_full_path = images_dir / frontal_path
            # Try with different extensions if needed
            if not frontal_full_path.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt_path = images_dir / (frontal_path.rsplit(".", 1)[0] + ext)
                    if alt_path.exists():
                        frontal_path = alt_path.name
                        break
                else:
                    raise FileNotFoundError(
                        f"Image file not found: {frontal_full_path}\n"
                        f"Study UID: {uid}\n"
                        f"Please verify image files are extracted correctly."
                    )
        
        if lateral_path:
            lateral_full_path = images_dir / lateral_path
            if not lateral_full_path.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt_path = images_dir / (lateral_path.rsplit(".", 1)[0] + ext)
                    if alt_path.exists():
                        lateral_path = alt_path.name
                        break
                else:
                    raise FileNotFoundError(
                        f"Image file not found: {lateral_full_path}\n"
                        f"Study UID: {uid}\n"
                        f"Please verify image files are extracted correctly."
                    )
        
        # Create Study object
        study = Study(
            images={
                "frontal": frontal_path,
                "lateral": lateral_path
            },
            indication=indication,
            findings=findings,
            impression=impression,
            labels=None,
            source="iu"
        )
        
        studies.append(study)
    
    if len(studies) == 0:
        raise ValueError("No valid studies found after processing. Check data integrity.")
    
    return studies


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
        data_dir: Optional[str] = None,
        split: str = "train",
        image_size: int = 224,
        max_seq_length: int = 256,
        tokenizer_name: str = "gpt2",
        transform: Optional[transforms.Compose] = None,
        return_multi_view: bool = True,
        target_type: str = "report",  # "findings", "impression", or "report"
        studies: Optional[List[Study]] = None,
    ):
        """
        Args:
            data_dir: Path to IU-Xray dataset (defaults to config.IU_DIR)
            split: One of "train", "val", "test"
            image_size: Image resize dimension
            max_seq_length: Maximum sequence length for tokenization
            tokenizer_name: HuggingFace tokenizer name
            transform: Optional custom transforms
            return_multi_view: Whether to return multiple views per study
            target_type: Which text to use as target
            studies: Optional pre-loaded list of Study objects
        """
        if data_dir is None:
            data_dir = Path(IU_DIR)
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.return_multi_view = return_multi_view
        self.target_type = target_type
        
        # Load studies
        if studies is None:
            all_studies = load_iu_xray_studies(data_dir)
        else:
            all_studies = studies
        
        # Create train/val/test splits
        self.studies = self._create_splits(all_studies)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Image transforms
        self.transform = transform or self._default_transforms(split == "train")
        
        # Image directory
        self.image_dir = self._find_image_dir()
        
        # CheXpert labels from indication/findings
        self.chexpert_labels = self._extract_chexpert_labels()
    
    def _find_image_dir(self) -> Path:
        """Find the image directory (including nested structures)"""
        # Check direct directories
        candidates = ["images", "Images", "images_normalized"]
        for candidate in candidates:
            path = self.data_dir / candidate
            if path.exists() and path.is_dir():
                # Check if it has images
                if len(list(path.glob("*.png"))) + len(list(path.glob("*.jpg"))) + len(list(path.glob("*.jpeg"))) > 0:
                    return path
                # Check nested directories
                for nested in ["images_normalized", "Images"]:
                    nested_path = path / nested
                    if nested_path.exists() and nested_path.is_dir():
                        if len(list(nested_path.glob("*.png"))) + len(list(nested_path.glob("*.jpg"))) + len(list(nested_path.glob("*.jpeg"))) > 0:
                            return nested_path
        raise FileNotFoundError(f"No image directory found in {self.data_dir}")
    
    def _create_splits(self, all_studies: List[Study]) -> List[Study]:
        """Create train/val/test splits"""
        # Shuffle with fixed seed
        np.random.seed(42)
        indices = np.arange(len(all_studies))
        np.random.shuffle(indices)
        
        n_total = len(all_studies)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if self.split == "train":
            split_indices = indices[:n_train]
        elif self.split == "val":
            split_indices = indices[n_train:n_train + n_val]
        else:  # test
            split_indices = indices[n_train + n_val:]
        
        return [all_studies[i] for i in split_indices]
    
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
    
    def _extract_chexpert_labels(self) -> Dict[int, torch.Tensor]:
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
        
        for idx, study in enumerate(self.studies):
            text = (study.findings + " " + study.impression).lower()
            
            label_vec = torch.zeros(len(labels))
            
            for i, label in enumerate(labels):
                if label in keywords:
                    for keyword in keywords[label]:
                        if keyword in text:
                            label_vec[i] = 1.0
                            break
            
            study_labels[idx] = label_vec
        
        return study_labels
    
    def _load_image(self, filename: Optional[str]) -> torch.Tensor:
        """Load and transform a single image"""
        if filename is None:
            return torch.zeros(3, self.image_size, self.image_size)
        
        # Handle various filename formats
        image_path = self.image_dir / filename
        
        # Try with different extensions if needed
        if not image_path.exists():
            base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
            for ext in [".png", ".jpg", ".jpeg"]:
                alt_path = self.image_dir / (base_name + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}\n"
                f"Expected in: {self.image_dir.absolute()}"
            )
        
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
    
    def _get_text_target(self, study: Study) -> str:
        """Get target text based on target_type"""
        if self.target_type == "findings":
            return study.findings
        elif self.target_type == "impression":
            return study.impression
        else:  # "report"
            parts = []
            if study.findings:
                parts.append(f"FINDINGS: {study.findings}")
            if study.impression:
                parts.append(f"IMPRESSION: {study.impression}")
            return " ".join(parts)
    
    def __len__(self) -> int:
        return len(self.studies)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        study = self.studies[idx]
        
        # Load images
        if self.return_multi_view:
            frontal_img = self._load_image(study.images["frontal"])
            lateral_img = self._load_image(study.images["lateral"])
            images = torch.stack([frontal_img, lateral_img])  # [2, 3, H, W]
        else:
            frontal_img = self._load_image(study.images["frontal"])
            images = frontal_img.unsqueeze(0)  # [1, 3, H, W]
        
        # Get target text
        target_text = self._get_text_target(study)
        
        # Tokenize
        encoding = self.tokenizer(
            target_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get clinical indication
        indication = study.indication if study.indication else "No clinical indication provided."
        indication_encoding = self.tokenizer(
            indication,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get CheXpert labels
        labels = self.chexpert_labels.get(idx, torch.zeros(14))
        
        return {
            "uid": idx,  # Use index as uid for now
            "images": images,  # [num_views, 3, H, W]
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "indication_ids": indication_encoding["input_ids"].squeeze(0),
            "indication_mask": indication_encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "target_text": target_text,
        }


def create_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    max_seq_length: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders
    
    Args:
        data_dir: Path to dataset (defaults to config.IU_DIR)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        max_seq_length: Maximum sequence length
    
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
        data_dir=None,  # Use default from config
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
