"""
Data preparation script for custom ShapeNet format
Helps organize and validate your ShapeNet data in the format: category_id/object_id.txt
"""

import os
import argparse
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split


def validate_txt_file(file_path):
    """Validate that a txt file contains valid point cloud data"""
    try:
        data = np.loadtxt(file_path)
        
        # Check if file has at least 3 columns (x, y, z)
        if data.shape[1] < 3:
            return False, f"Insufficient columns: {data.shape[1]} (need at least 3)"
        
        # Check if coordinates are valid
        if not np.all(np.isfinite(data[:, :3])):
            return False, "Contains invalid coordinates (NaN or inf)"
        
        # Check if file has enough points
        if len(data) < 100:
            return False, f"Too few points: {len(data)} (need at least 100)"
        
        # Check labels if they exist
        if data.shape[1] >= 4:
            labels = data[:, 3].astype(int)
            if labels.min() < 1:
                return False, "Labels should be 1-indexed"
            if labels.max() > 20:  # Reasonable upper bound
                return False, f"Labels seem too high: max={labels.max()}"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def scan_shapenet_directory(root_dir):
    """Scan ShapeNet directory and collect information"""
    print(f"Scanning directory: {root_dir}")
    
    categories = {}
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    # Find all category directories
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            category_id = item
            txt_files = glob(os.path.join(item_path, "*.txt"))
            
            if txt_files:
                categories[category_id] = {
                    'path': item_path,
                    'files': txt_files,
                    'count': len(txt_files),
                    'valid': 0,
                    'invalid': 0
                }
                
                print(f"  Category {category_id}: {len(txt_files)} files")
                
                # Validate files
                for txt_file in tqdm(txt_files, desc=f"Validating {category_id}"):
                    is_valid, message = validate_txt_file(txt_file)
                    total_files += 1
                    
                    if is_valid:
                        valid_files += 1
                        categories[category_id]['valid'] += 1
                    else:
                        invalid_files.append((txt_file, message))
                        categories[category_id]['invalid'] += 1
    
    print(f"\nScan complete:")
    print(f"  Total categories: {len(categories)}")
    print(f"  Total files: {total_files}")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    return categories, invalid_files


def create_train_test_splits(categories, root_dir, train_ratio=0.8, min_files_per_category=10):
    """Create train/test splits for the dataset"""
    print(f"\nCreating train/test splits (train ratio: {train_ratio})...")
    
    train_files = []
    test_files = []
    
    for category_id, info in categories.items():
        valid_files = []
        
        # Filter out invalid files
        for txt_file in info['files']:
            is_valid, _ = validate_txt_file(txt_file)
            if is_valid:
                # Convert to relative path
                rel_path = os.path.relpath(txt_file, root_dir)
                valid_files.append(rel_path)
        
        if len(valid_files) < min_files_per_category:
            print(f"  Skipping {category_id}: only {len(valid_files)} valid files")
            continue
        
        # Split files for this category
        category_train, category_test = train_test_split(
            valid_files, 
            train_size=train_ratio,
            random_state=42
        )
        
        train_files.extend(category_train)
        test_files.extend(category_test)
        
        print(f"  {category_id}: {len(category_train)} train, {len(category_test)} test")
    
    # Save splits
    splits_dir = os.path.join(root_dir, 'train_test_split')
    os.makedirs(splits_dir, exist_ok=True)
    
    splits = {
        'train': train_files,
        'test': test_files
    }
    
    splits_file = os.path.join(splits_dir, 'file_splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {splits_file}")
    print(f"  Training files: {len(train_files)}")
    print(f"  Test files: {len(test_files)}")
    
    return splits


def create_category_mapping(categories, root_dir):
    """Create category mapping file"""
    # Known ShapeNet categories
    shapenet_categories = {
        '02691156': 'airplane',
        '02773838': 'bag',
        '02954340': 'cap',
        '02958343': 'car',
        '03001627': 'chair',
        '03261776': 'earphone',
        '03467517': 'guitar',
        '03624134': 'knife',
        '03636649': 'lamp',
        '03642806': 'laptop',
        '03790512': 'motorbike',
        '03797390': 'mug',
        '03948459': 'pistol',
        '04099429': 'rocket',
        '04225987': 'skateboard',
        '04379243': 'table'
    }
    
    category_mapping = {}
    for category_id in categories.keys():
        category_name = shapenet_categories.get(category_id, f'unknown_{category_id}')
        category_mapping[category_id] = category_name
    
    # Save mapping
    mapping_file = os.path.join(root_dir, 'category_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(category_mapping, f, indent=2)
    
    print(f"\nCategory mapping saved to: {mapping_file}")
    for cat_id, cat_name in category_mapping.items():
        print(f"  {cat_id}: {cat_name}")


def analyze_segmentation_classes(categories):
    """Analyze segmentation classes in the dataset"""
    print("\nAnalyzing segmentation classes...")
    
    category_classes = {}
    
    for category_id, info in categories.items():
        max_label = 0
        label_counts = {}
        
        # Sample some files to analyze labels
        sample_files = info['files'][:min(50, len(info['files']))]
        
        for txt_file in sample_files:
            try:
                data = np.loadtxt(txt_file)
                if data.shape[1] >= 4:
                    labels = data[:, 3].astype(int)
                    max_label = max(max_label, labels.max())
                    
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        label_counts[label] = label_counts.get(label, 0) + count
                        
            except Exception:
                continue
        
        category_classes[category_id] = {
            'max_label': max_label,
            'num_classes': max_label,
            'label_distribution': label_counts
        }
        
        print(f"  {category_id}: {max_label} classes, distribution: {dict(sorted(label_counts.items()))}")
    
    return category_classes


def fix_invalid_files(invalid_files, backup_dir=None):
    """Attempt to fix or move invalid files"""
    if not invalid_files:
        print("No invalid files to fix")
        return
    
    print(f"\nProcessing {len(invalid_files)} invalid files...")
    
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
    
    fixed_count = 0
    moved_count = 0
    
    for file_path, error_msg in invalid_files:
        try:
            # Try simple fixes
            if "Too few points" in error_msg:
                # Skip files with too few points
                if backup_dir:
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    shutil.move(file_path, backup_path)
                    moved_count += 1
                continue
            
            # Try to reload and fix data
            data = np.loadtxt(file_path)
            
            # Remove invalid coordinates
            if data.shape[1] >= 3:
                valid_mask = np.all(np.isfinite(data[:, :3]), axis=1)
                if np.any(valid_mask):
                    data = data[valid_mask]
                    
                    # Fix labels if needed
                    if data.shape[1] >= 4:
                        labels = data[:, 3].astype(int)
                        labels = np.clip(labels, 1, 20)  # Clamp to reasonable range
                        data[:, 3] = labels
                    
                    # Save fixed file
                    if len(data) >= 100:  # Minimum points threshold
                        np.savetxt(file_path, data, fmt='%.6f')
                        fixed_count += 1
                    else:
                        if backup_dir:
                            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                            shutil.move(file_path, backup_path)
                            moved_count += 1
                            
        except Exception as e:
            print(f"  Could not fix {file_path}: {e}")
    
    print(f"  Fixed: {fixed_count} files")
    print(f"  Moved to backup: {moved_count} files")


def main():
    parser = argparse.ArgumentParser(description='Prepare ShapeNet data for PointNet training')
    parser.add_argument('--dataset', type=str, required=True, help='Path to ShapeNet dataset')
    parser.add_argument('--create_splits', action='store_true', help='Create train/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
    parser.add_argument('--fix_invalid', action='store_true', help='Attempt to fix invalid files')
    parser.add_argument('--backup_dir', type=str, help='Directory to backup invalid files')
    parser.add_argument('--validate_only', action='store_true', help='Only validate files, don\'t create splits')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory does not exist: {args.dataset}")
        return
    
    print("=== ShapeNet Data Preparation ===")
    
    # Scan and validate dataset
    categories, invalid_files = scan_shapenet_directory(args.dataset)
    
    if not categories:
        print("No valid categories found!")
        return
    
    # Fix invalid files if requested
    if args.fix_invalid:
        fix_invalid_files(invalid_files, args.backup_dir)
        # Re-scan after fixing
        categories, invalid_files = scan_shapenet_directory(args.dataset)
    
    # Create category mapping
    create_category_mapping(categories, args.dataset)
    
    # Analyze segmentation classes
    analyze_segmentation_classes(categories)
    
    # Create train/test splits
    if args.create_splits and not args.validate_only:
        create_train_test_splits(categories, args.dataset, args.train_ratio)
    
    print("\n=== Data Preparation Complete ===")
    print(f"Your data is ready for training!")
    print(f"Use: python3 utils/train_custom_segmentation.py --dataset {args.dataset}")


if __name__ == '__main__':
    main()