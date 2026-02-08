"""
IU-Xray Dataset Verification and Preprocessing Scripts
Automatically handles zip file extraction if dataset is not already extracted.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import IU_DIR, IU_REPORTS_CSV, IU_PROJECTIONS_CSV, IU_IMAGES_DIR, DATA_ROOT


def find_zip_files(data_root: Optional[str] = None) -> List[Path]:
    """
    Find zip files in the data directory that might contain the IU-Xray dataset.
    
    Args:
        data_root: Root data directory (defaults to config.DATA_ROOT)
    
    Returns:
        List of zip file paths
    """
    if data_root is None:
        data_root = Path(DATA_ROOT)
    else:
        data_root = Path(data_root)
    
    if not data_root.exists():
        return []
    
    # Look for common zip file patterns
    zip_patterns = [
        "*.zip",
        "archive*.zip",
        "*iu*.zip",
        "*xray*.zip",
        "*indiana*.zip"
    ]
    
    zip_files = []
    for pattern in zip_patterns:
        zip_files.extend(data_root.glob(pattern))
        # Also check in subdirectories (one level deep)
        for subdir in data_root.iterdir():
            if subdir.is_dir():
                zip_files.extend(subdir.glob(pattern))
    
    # Remove duplicates and sort
    zip_files = sorted(set(zip_files))
    return zip_files


def is_dataset_extracted(data_dir: Path) -> bool:
    """
    Check if the dataset is already extracted (CSV files exist).
    
    Args:
        data_dir: Dataset directory to check
    
    Returns:
        True if dataset appears to be extracted
    """
    required_files = [
        data_dir / "indiana_reports.csv",
        data_dir / "indiana_projections.csv"
    ]
    
    # Check if at least one CSV exists
    csv_exists = any(f.exists() for f in required_files)
    
    # Check if images directory exists
    image_dirs = ["images", "Images", "images_normalized"]
    images_exist = any((data_dir / img_dir).exists() for img_dir in image_dirs)
    
    return csv_exists or images_exist


def extract_zip_file(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file to the target directory.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    
    Returns:
        True if extraction successful
    """
    if not zip_path.exists():
        return False
    
    print(f"\n[ZIP] Found zip file: {zip_path.name}")
    print(f"[ZIP] Extracting to: {extract_to.absolute()}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(info.file_size for info in zip_ref.infolist())
            
            # Extract with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for member in zip_ref.infolist():
                    zip_ref.extract(member, extract_to)
                    pbar.update(member.file_size)
        
        print(f"[OK] Extraction complete!")
        return True
    except zipfile.BadZipFile:
        print(f"[ERROR] {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"[ERROR] Error extracting {zip_path}: {e}")
        return False


def ensure_dataset_extracted(data_dir: Optional[str] = None, auto_extract: bool = True) -> Path:
    """
    Ensure the dataset is extracted. If not, try to extract from zip files.
    
    This function:
    1. Checks if dataset is already extracted
    2. If not, looks for zip files in data directory
    3. Automatically extracts the first suitable zip file found
    4. Returns the dataset directory path
    
    Args:
        data_dir: Optional override for dataset directory (defaults to IU_DIR from config)
        auto_extract: If True, automatically extract zip files if found
    
    Returns:
        Path to dataset directory
    
    Raises:
        FileNotFoundError: If dataset not extracted and no zip file found
    """
    if data_dir is None:
        data_dir = Path(IU_DIR)
    else:
        data_dir = Path(data_dir)
    
    # Check if already extracted
    if is_dataset_extracted(data_dir):
        print(f"[OK] Dataset already extracted at: {data_dir.absolute()}")
        return data_dir
    
    print(f"Dataset not found at: {data_dir.absolute()}")
    
    if not auto_extract:
        raise FileNotFoundError(
            f"Dataset not extracted at {data_dir.absolute()}\n"
            f"Please extract the dataset manually or set auto_extract=True"
        )
    
    # Look for zip files
    print("\n[SEARCH] Searching for zip files...")
    data_root = data_dir.parent if data_dir.name == "iu_xray" else Path(DATA_ROOT)
    zip_files = find_zip_files(data_root)
    
    if not zip_files:
        raise FileNotFoundError(
            f"Dataset not extracted and no zip files found.\n"
            f"Expected dataset location: {data_dir.absolute()}\n"
            f"Searched for zip files in: {data_root.absolute()}\n"
            f"\nPlease either:\n"
            f"  1. Extract the dataset to: {data_dir.absolute()}\n"
            f"  2. Place a zip file containing the dataset in: {data_root.absolute()}"
        )
    
    print(f"Found {len(zip_files)} zip file(s):")
    for zf in zip_files:
        size_mb = zf.stat().st_size / (1024 * 1024)
        print(f"  - {zf.name} ({size_mb:.1f} MB)")
    
    # Try extracting the first (usually largest) zip file
    # Sort by size (largest first) as IU-Xray is ~13 GB
    zip_files_sorted = sorted(zip_files, key=lambda x: x.stat().st_size, reverse=True)
    
    for zip_file in zip_files_sorted:
        print(f"\n[EXTRACT] Attempting to extract: {zip_file.name}")
        
        # Extract to a temporary location first, then move if successful
        temp_extract = data_dir.parent / f"{data_dir.name}_temp"
        
        if extract_zip_file(zip_file, temp_extract):
            # Check if extraction was successful
            if is_dataset_extracted(temp_extract):
                # Move contents to final location
                print(f"\n[ORGANIZE] Organizing extracted files...")
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Move files from temp to final location
                for item in temp_extract.iterdir():
                    dest = data_dir / item.name
                    if dest.exists():
                        # If destination exists, merge directories
                        if item.is_dir():
                            import shutil
                            for subitem in item.iterdir():
                                shutil.move(str(subitem), str(dest / subitem.name))
                        else:
                            item.unlink()  # Skip if file exists
                    else:
                        import shutil
                        shutil.move(str(item), str(dest))
                
                # Clean up temp directory
                if temp_extract.exists():
                    import shutil
                    shutil.rmtree(temp_extract)
                
                # Verify extraction
                if is_dataset_extracted(data_dir):
                    print(f"[OK] Dataset successfully extracted to: {data_dir.absolute()}")
                    return data_dir
                else:
                    print(f"[WARNING] Extraction completed but dataset structure not recognized")
                    print(f"  Please verify files are in: {data_dir.absolute()}")
            else:
                print(f"[WARNING] Extracted but dataset structure not found in zip")
                # Try extracting directly to data_dir
                if extract_zip_file(zip_file, data_dir):
                    if is_dataset_extracted(data_dir):
                        return data_dir
    
    # If we get here, extraction didn't work
    raise FileNotFoundError(
        f"Could not extract dataset from zip files.\n"
        f"Please manually extract the dataset to: {data_dir.absolute()}\n"
        f"Expected structure:\n"
        f"  {data_dir}/indiana_reports.csv\n"
        f"  {data_dir}/indiana_projections.csv\n"
        f"  {data_dir}/images/"
    )


def verify_iu_xray(data_dir: Optional[str] = None, auto_extract: bool = True) -> bool:
    """
    Verify IU-Xray dataset integrity.
    Automatically extracts from zip if needed.
    Raises FileNotFoundError if required files are missing.
    
    Args:
        data_dir: Optional override for dataset directory (defaults to IU_DIR from config)
        auto_extract: If True, automatically extract zip files if dataset not found
    
    Returns:
        True if all required files exist
    
    Raises:
        FileNotFoundError: If any required file is missing
    """
    # Ensure dataset is extracted first
    data_dir = ensure_dataset_extracted(data_dir, auto_extract=auto_extract)
    
    print("=" * 60)
    print("IU-Xray Dataset Verification")
    print("=" * 60)
    print(f"Checking dataset at: {data_dir.absolute()}")
    print()
    
    # Check required CSV files
    required_files = {
        "indiana_reports.csv": data_dir / "indiana_reports.csv",
        "indiana_projections.csv": data_dir / "indiana_projections.csv"
    }
    
    missing_files = []
    for name, path in required_files.items():
        if path.exists():
            print(f"  [OK] {name} found")
        else:
            print(f"  [MISSING] {name} MISSING")
            missing_files.append(str(path))
    
    if missing_files:
        raise FileNotFoundError(
            f"Required CSV files not found:\n" + "\n".join(f"  - {f}" for f in missing_files) +
            f"\n\nPlease ensure the IU-Xray dataset is extracted to: {data_dir.absolute()}"
        )
    
    # Check for images directory (including nested structures)
    image_dirs = ["images", "Images", "images_normalized"]
    images_found = False
    images_path = None
    
    # First check direct directories
    for img_dir in image_dirs:
        img_path = data_dir / img_dir
        if img_path.exists() and img_path.is_dir():
            # Check for images in this directory
            num_images = len(list(img_path.glob("*.png"))) + len(list(img_path.glob("*.jpg"))) + len(list(img_path.glob("*.jpeg")))
            if num_images > 0:
                print(f"  [OK] {img_dir}/ found with {num_images} images")
                images_found = True
                images_path = img_path
                break
            # Also check nested directories (e.g., images/images_normalized/)
            for nested_dir in image_dirs:
                nested_path = img_path / nested_dir
                if nested_path.exists() and nested_path.is_dir():
                    num_images = len(list(nested_path.glob("*.png"))) + len(list(nested_path.glob("*.jpg"))) + len(list(nested_path.glob("*.jpeg")))
                    if num_images > 0:
                        print(f"  [OK] {img_dir}/{nested_dir}/ found with {num_images} images")
                        images_found = True
                        images_path = nested_path
                        break
            if images_found:
                break
    
    if not images_found:
        raise FileNotFoundError(
            f"No image directory found in {data_dir.absolute()}\n"
            f"Expected one of: {', '.join(image_dirs)}\n"
            f"Please ensure images are extracted to: {data_dir.absolute()}/images/"
        )
    
    # Load and check data
    print("\nLoading CSV files...")
    try:
        proj_df = pd.read_csv(required_files["indiana_projections.csv"])
        print(f"  Projections: {len(proj_df)} rows")
        print(f"    Columns: {list(proj_df.columns)}")
        
        reports_df = pd.read_csv(required_files["indiana_reports.csv"])
        print(f"  Reports: {len(reports_df)} rows")
        print(f"    Columns: {list(reports_df.columns)}")
        
        # Check for uid column
        if "uid" not in proj_df.columns:
            raise ValueError("indiana_projections.csv missing 'uid' column")
        if "uid" not in reports_df.columns:
            raise ValueError("indiana_reports.csv missing 'uid' column")
        
        # Check merge compatibility
        common_uids = set(proj_df["uid"]).intersection(set(reports_df["uid"]))
        print(f"\n  Common UIDs: {len(common_uids)} studies")
        
        if len(common_uids) == 0:
            raise ValueError("No common UIDs found between projections and reports CSV files")
        
    except Exception as e:
        raise RuntimeError(f"Error reading CSV files: {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Dataset verification complete!")
    print("=" * 60)
    
    return True


def preprocess_iu_xray(data_dir: Optional[str] = None, auto_extract: bool = True) -> pd.DataFrame:
    """
    Preprocess IU-Xray data:
    - Merge projections and reports
    - Extract findings and impressions
    - Create train/val/test splits
    - Save processed data
    Automatically extracts from zip if needed.
    
    Args:
        data_dir: Optional override for dataset directory (defaults to IU_DIR from config)
        auto_extract: If True, automatically extract zip files if dataset not found
    
    Returns:
        Merged and processed DataFrame
    
    Raises:
        FileNotFoundError: If required CSV files are missing
    """
    print("\n" + "=" * 60)
    print("Preprocessing IU-Xray dataset...")
    print("=" * 60)
    
    # Ensure dataset is extracted and verify
    data_dir = ensure_dataset_extracted(data_dir, auto_extract=auto_extract)
    verify_iu_xray(data_dir, auto_extract=False)
    
    # Load data
    projections_path = data_dir / "indiana_projections.csv"
    reports_path = data_dir / "indiana_reports.csv"
    
    if not projections_path.exists():
        raise FileNotFoundError(f"Missing required file: {projections_path}")
    if not reports_path.exists():
        raise FileNotFoundError(f"Missing required file: {reports_path}")
    
    print("\nLoading CSV files...")
    projections = pd.read_csv(projections_path)
    reports = pd.read_csv(reports_path)
    
    # Merge on uid
    merged = projections.merge(reports, on="uid", how="inner")
    print(f"Merged dataset: {len(merged)} image-report pairs")
    
    if len(merged) == 0:
        raise ValueError("No matching UIDs found between projections and reports. Check data integrity.")
    
    # Clean text fields
    text_columns = ["findings", "impression", "indication"]
    for col in text_columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")
            merged[col] = merged[col].str.strip()
        else:
            merged[col] = ""
    
    # Filter out rows with empty findings AND impression
    valid_mask = (merged["findings"].str.len() > 0) | (merged["impression"].str.len() > 0)
    merged = merged[valid_mask].reset_index(drop=True)
    print(f"After filtering empty reports: {len(merged)} samples")
    
    if len(merged) == 0:
        raise ValueError("No valid reports found after filtering. All reports are empty.")
    
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
    
    parser = argparse.ArgumentParser(description="Verify and preprocess IU-Xray dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing IU-Xray dataset (defaults to config.IU_DIR)")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess dataset and create splits")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset integrity")
    parser.add_argument("--no-auto-extract", action="store_true",
                        help="Disable automatic zip extraction")
    
    args = parser.parse_args()
    
    auto_extract = not args.no_auto_extract
    
    if args.verify or not args.preprocess:
        verify_iu_xray(args.data_dir, auto_extract=auto_extract)
        
    if args.preprocess:
        preprocess_iu_xray(args.data_dir, auto_extract=auto_extract)
