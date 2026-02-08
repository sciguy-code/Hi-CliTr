"""
Dataset Sanity Check Script
Loads IU-Xray dataset and verifies integrity
"""

import sys
from pathlib import Path
from typing import Tuple, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import load_iu_xray_studies, Study
from config import IU_DIR
import json


def print_study_sample(study: Study, index: int = 0):
    """Print a formatted Study sample"""
    print("\n" + "=" * 80)
    print(f"STUDY SAMPLE #{index + 1}")
    print("=" * 80)
    
    # Convert Study to dict for pretty printing
    study_dict = study.to_dict()
    
    # Handle images separately
    print("\n[IMAGES]")
    print(f"  Frontal: {study_dict['images']['frontal']}")
    print(f"  Lateral: {study_dict['images']['lateral']}")
    
    print("\n[CLINICAL INDICATION]")
    indication = study_dict['indication'] if study_dict['indication'] else "(empty)"
    print(f"  {indication}")
    
    print("\n[FINDINGS]")
    findings = study_dict['findings'] if study_dict['findings'] else "(empty)"
    # Truncate if too long
    if len(findings) > 500:
        findings = findings[:500] + "..."
    print(f"  {findings}")
    
    print("\n[IMPRESSION]")
    impression = study_dict['impression'] if study_dict['impression'] else "(empty)"
    # Truncate if too long
    if len(impression) > 500:
        impression = impression[:500] + "..."
    print(f"  {impression}")
    
    print(f"\n[SOURCE] {study_dict['source']}")
    print(f"[LABELS] {study_dict['labels']}")
    
    print("=" * 80)


def verify_image_paths(study: Study, images_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify that image paths in study exist on disk
    
    Returns:
        (all_exist, missing_files)
    """
    missing = []
    
    if study.images["frontal"]:
        frontal_path = images_dir / study.images["frontal"]
        if not frontal_path.exists():
            # Try with different extensions
            base = study.images["frontal"].rsplit(".", 1)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                if (images_dir / (base + ext)).exists():
                    found = True
                    break
            if not found:
                missing.append(str(frontal_path))
    
    if study.images["lateral"]:
        lateral_path = images_dir / study.images["lateral"]
        if not lateral_path.exists():
            # Try with different extensions
            base = study.images["lateral"].rsplit(".", 1)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                if (images_dir / (base + ext)).exists():
                    found = True
                    break
            if not found:
                missing.append(str(lateral_path))
    
    return len(missing) == 0, missing


def sanity_check(data_dir: str = None):
    """
    Perform comprehensive dataset sanity check
    
    Args:
        data_dir: Optional override for dataset directory
    """
    print("=" * 80)
    print("IU-XRAY DATASET SANITY CHECK")
    print("=" * 80)
    
    if data_dir is None:
        data_dir = Path(IU_DIR)
    else:
        data_dir = Path(data_dir)
    
    print(f"\nDataset directory: {data_dir.absolute()}")
    
    # Load studies (with auto-extraction)
    print("\n[LOADING] Loading IU-Xray dataset...")
    try:
        studies = load_iu_xray_studies(data_dir, auto_extract=True)
        print(f"[OK] Successfully loaded {len(studies)} studies")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset")
        print(f"  {type(e).__name__}: {e}")
        return False
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nTotal studies: {len(studies)}")
    
    # Count studies with frontal/lateral images
    frontal_count = sum(1 for s in studies if s.images["frontal"] is not None)
    lateral_count = sum(1 for s in studies if s.images["lateral"] is not None)
    both_count = sum(1 for s in studies if s.images["frontal"] and s.images["lateral"])
    
    print(f"Studies with frontal images: {frontal_count}")
    print(f"Studies with lateral images: {lateral_count}")
    print(f"Studies with both views: {both_count}")
    
    # Count studies with text
    indication_count = sum(1 for s in studies if s.indication.strip())
    findings_count = sum(1 for s in studies if s.findings.strip())
    impression_count = sum(1 for s in studies if s.impression.strip())
    
    print(f"\nStudies with indication: {indication_count}")
    print(f"Studies with findings: {findings_count}")
    print(f"Studies with impression: {impression_count}")
    
    # Find images directory (use the same logic as dataset loader)
    # The studies were loaded successfully, so we know images_dir was found
    # Let's find it the same way
    images_dir = None
    for base_name in ["images", "Images"]:
        base_path = data_dir / base_name
        if base_path.exists() and base_path.is_dir():
            # Check nested directories first
            for nested_name in ["images_normalized", "Images", "images"]:
                nested_path = base_path / nested_name
                if nested_path.exists() and nested_path.is_dir():
                    num_images = len(list(nested_path.glob("*.png"))) + len(list(nested_path.glob("*.jpg"))) + len(list(nested_path.glob("*.jpeg")))
                    if num_images > 0:
                        images_dir = nested_path
                        break
            # Also check if base_path itself has images
            if images_dir is None:
                num_images = len(list(base_path.glob("*.png"))) + len(list(base_path.glob("*.jpg"))) + len(list(base_path.glob("*.jpeg")))
                if num_images > 0:
                    images_dir = base_path
    
    # Try direct directories
    if images_dir is None:
        for alt_name in ["Images", "images_normalized"]:
            alt_path = data_dir / alt_name
            if alt_path.exists() and alt_path.is_dir():
                num_images = len(list(alt_path.glob("*.png"))) + len(list(alt_path.glob("*.jpg"))) + len(list(alt_path.glob("*.jpeg")))
                if num_images > 0:
                    images_dir = alt_path
                    break
    
    if images_dir is None:
        images_dir = data_dir / "images"  # Fallback
    
    # Verify image paths
    print("\n" + "=" * 80)
    print("IMAGE PATH VERIFICATION")
    print("=" * 80)
    print(f"\nChecking images in: {images_dir.absolute()}")
    
    missing_count = 0
    verified_count = 0
    
    for i, study in enumerate(studies):
        all_exist, missing = verify_image_paths(study, images_dir)
        if all_exist:
            verified_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing
                print(f"\n[WARNING] Study #{i+1} missing images:")
                for m in missing:
                    print(f"  - {m}")
    
    print(f"\n[OK] Verified: {verified_count}/{len(studies)} studies have all images")
    if missing_count > 0:
        print(f"[WARNING] Missing: {missing_count}/{len(studies)} studies have missing images")
    
    # Print sample study
    if len(studies) > 0:
        print_study_sample(studies[0], index=0)
        
        # Print another sample if available
        if len(studies) > 1:
            # Find a study with both views if possible
            for i, s in enumerate(studies[1:], 1):
                if s.images["frontal"] and s.images["lateral"]:
                    print_study_sample(s, index=i)
                    break
    
    # Summary
    print("\n" + "=" * 80)
    print("SANITY CHECK SUMMARY")
    print("=" * 80)
    
    all_good = (
        len(studies) > 0 and
        verified_count == len(studies) and
        findings_count > 0 and
        impression_count > 0
    )
    
    if all_good:
        print("\n[OK] All checks passed! Dataset is ready for training.")
    else:
        print("\n[WARNING] Some issues found:")
        if len(studies) == 0:
            print("  - No studies loaded")
        if verified_count < len(studies):
            print(f"  - {missing_count} studies have missing images")
        if findings_count == 0:
            print("  - No studies have findings text")
        if impression_count == 0:
            print("  - No studies have impression text")
    
    print("=" * 80)
    
    return all_good


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IU-Xray Dataset Sanity Check")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing IU-Xray dataset (defaults to config.IU_DIR)")
    
    args = parser.parse_args()
    
    success = sanity_check(args.data_dir)
    sys.exit(0 if success else 1)
