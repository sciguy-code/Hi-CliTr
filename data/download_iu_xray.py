"""
Data Download and Preprocessing Scripts for IU-Xray Dataset
"""

import os
import kaggle
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def download_iu_xray(data_dir: str = "./data/iu_xray"):
    """
    Download IU-Xray dataset from Kaggle
    
    Dataset: https://www.kaggle.com/raddar/chest-xrays-indiana-university
    
    Args:
        data_dir: Directory to save the dataset
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("IU-Xray Dataset Downloader")
    print("=" * 60)
    
    # Check if already downloaded
    if (data_dir / "indiana_projections.csv").exists() and (data_dir / "indiana_reports.csv").exists():
        print(f"Dataset already exists at {data_dir}")
        return
    
    print("\nDownloading from Kaggle...")
    print("Note: Requires Kaggle API credentials in ~/.kaggle/kaggle.json")
    print("Set up: https://www.kaggle.com/docs/api#authentication")
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            'raddar/chest-xrays-indiana-university',
            path=str(data_dir),
            unzip=True
        )
        print(f"\nDataset downloaded to {data_dir}")
        
    except Exception as e:
        print(f"\nKaggle API error: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/raddar/chest-xrays-indiana-university")
        print("2. Download and extract to:", data_dir)
        print("3. Ensure these files exist:")
        print("   - indiana_projections.csv")
        print("   - indiana_reports.csv")
        print("   - images/ folder with PNG files")
        return
    
    # Verify download
    verify_iu_xray(data_dir)


def verify_iu_xray(data_dir: str = "./data/iu_xray"):
    """Verify IU-Xray dataset integrity"""
    data_dir = Path(data_dir)
    
    print("\nVerifying dataset...")
    
    required_files = [
        "indiana_projections.csv",
        "indiana_reports.csv"
    ]
    
    for f in required_files:
        if (data_dir / f).exists():
            print(f"  ✓ {f} found")
        else:
            print(f"  ✗ {f} MISSING")
            
    # Check for images
    image_dirs = ["images", "images_normalized"]
    images_found = False
    
    for img_dir in image_dirs:
        img_path = data_dir / img_dir
        if img_path.exists():
            num_images = len(list(img_path.glob("*.png")))
            print(f"  ✓ {img_dir}/ found with {num_images} images")
            images_found = True
            break
    
    if not images_found:
        print("  ✗ No image directory found")
        
    # Load and check data
    if (data_dir / "indiana_projections.csv").exists():
        proj_df = pd.read_csv(data_dir / "indiana_projections.csv")
        print(f"\nProjections: {len(proj_df)} rows")
        print(f"  Columns: {list(proj_df.columns)}")
        
    if (data_dir / "indiana_reports.csv").exists():
        reports_df = pd.read_csv(data_dir / "indiana_reports.csv")
        print(f"\nReports: {len(reports_df)} rows")
        print(f"  Columns: {list(reports_df.columns)}")
        

def preprocess_iu_xray(data_dir: str = "./data/iu_xray"):
    """
    Preprocess IU-Xray data:
    - Merge projections and reports
    - Extract findings and impressions
    - Create train/val/test splits
    - Save processed data
    """
    data_dir = Path(data_dir)
    
    print("\nPreprocessing IU-Xray dataset...")
    
    # Load data
    projections = pd.read_csv(data_dir / "indiana_projections.csv")
    reports = pd.read_csv(data_dir / "indiana_reports.csv")
    
    # Merge on uid
    merged = projections.merge(reports, on="uid", how="inner")
    print(f"Merged dataset: {len(merged)} image-report pairs")
    
    # Clean text fields
    text_columns = ["findings", "impression", "indication"]
    for col in text_columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")
            merged[col] = merged[col].str.strip()
    
    # Filter out rows with empty findings AND impression
    valid_mask = (merged["findings"].str.len() > 0) | (merged["impression"].str.len() > 0)
    merged = merged[valid_mask].reset_index(drop=True)
    print(f"After filtering empty reports: {len(merged)} samples")
    
    # Create combined target text
    def create_report(row):
        parts = []
        if row.get("findings", ""):
            parts.append(f"FINDINGS: {row['findings']}")
        if row.get("impression", ""):
            parts.append(f"IMPRESSION: {row['impression']}")
        return " ".join(parts)
    
    merged["report"] = merged.apply(create_report, axis=1)
    
    # Split by unique UIDs (patient/study level)
    unique_uids = merged["uid"].unique()
    n_total = len(unique_uids)
    
    # Shuffle with fixed seed
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(unique_uids)
    
    # 80/10/10 split
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_uids = set(unique_uids[:n_train])
    val_uids = set(unique_uids[n_train:n_train + n_val])
    test_uids = set(unique_uids[n_train + n_val:])
    
    # Assign splits
    merged["split"] = merged["uid"].apply(
        lambda x: "train" if x in train_uids else ("val" if x in val_uids else "test")
    )
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(merged[merged['split'] == 'train'])} samples")
    print(f"  Val: {len(merged[merged['split'] == 'val'])} samples")
    print(f"  Test: {len(merged[merged['split'] == 'test'])} samples")
    
    # Save processed data
    output_file = data_dir / "processed_data.csv"
    merged.to_csv(output_file, index=False)
    print(f"\nSaved processed data to: {output_file}")
    
    return merged


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and preprocess IU-Xray dataset")
    parser.add_argument("--data_dir", type=str, default="./data/iu_xray",
                        help="Directory to save dataset")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from Kaggle")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess dataset")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset integrity")
    
    args = parser.parse_args()
    
    if args.download:
        download_iu_xray(args.data_dir)
    
    if args.verify:
        verify_iu_xray(args.data_dir)
        
    if args.preprocess:
        preprocess_iu_xray(args.data_dir)
        
    # Default: do all
    if not any([args.download, args.verify, args.preprocess]):
        download_iu_xray(args.data_dir)
        preprocess_iu_xray(args.data_dir)
