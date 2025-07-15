"""
Custom ShapeNet dataset loader for format: 0300167/678988644.txt
"""

import torch.utils.data as data
import os
import numpy as np
import torch
from glob import glob


class CustomShapeNetDataset(data.Dataset):
    """
    Custom ShapeNet dataset loader for the format: category_id/object_id.txt
    where each .txt file contains point cloud data with labels
    """
    
    def __init__(self, root, npoints=2500, split='train', data_augmentation=True, 
                 class_choice=None, classification=False):
        """
        Args:
            root: Root directory containing category folders
            npoints: Number of points to sample
            split: 'train' or 'test' (you'll need to organize your data accordingly)
            data_augmentation: Whether to apply data augmentation
            class_choice: List of categories to include (e.g., ['03001627'])
            classification: Whether to use for classification (True) or segmentation (False)
        """
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.classification = classification
        
        # Scan for available categories
        self.categories = []
        self.files = []
        
        # If no class_choice specified, use all available categories
        if class_choice is None:
            category_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            class_choice = category_dirs
        
        # Build file list
        for category in class_choice:
            category_path = os.path.join(root, category)
            if os.path.exists(category_path):
                # Find all .txt files in this category
                txt_files = glob(os.path.join(category_path, "*.txt"))
                for txt_file in txt_files:
                    self.files.append((category, txt_file))
                    
        if len(self.files) == 0:
            raise RuntimeError(f"No .txt files found in {root}")
            
        # Create category to index mapping
        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(set(class_choice)))}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        # For segmentation, we need to determine number of classes
        # This will be set based on the actual data
        self.num_seg_classes = self._determine_seg_classes()
        
        print(f"Loaded {len(self.files)} files from {len(self.category_to_idx)} categories")
        print(f"Categories: {list(self.category_to_idx.keys())}")
        
    def _determine_seg_classes(self):
        """Determine the number of segmentation classes by scanning some files"""
        max_label = 0
        sample_size = min(100, len(self.files))  # Sample first 100 files
        
        for i in range(sample_size):
            try:
                _, file_path = self.files[i]
                data = np.loadtxt(file_path)
                
                if data.shape[1] >= 4:  # Has label column
                    labels = data[:, 3].astype(np.int32)
                    max_label = max(max_label, labels.max())
                    
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
                
        # Number of classes is max_label (assuming 1-indexed labels)
        return max_label if max_label > 0 else 1
    
    def _load_file(self, file_path):
        """Load point cloud data from text file"""
        try:
            data = np.loadtxt(file_path)
            
            if data.shape[1] < 3:
                raise ValueError(f"File {file_path} has insufficient columns (need at least 3 for xyz)")
                
            # Extract coordinates
            points = data[:, :3].astype(np.float32)
            
            # Extract labels if available
            labels = None
            if data.shape[1] >= 4:
                labels = data[:, 3].astype(np.int64)
                
            return points, labels
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def __getitem__(self, index):
        category, file_path = self.files[index]
        
        # Load point cloud and labels
        points, labels = self._load_file(file_path)
        
        if points is None:
            # Return dummy data if file couldn't be loaded
            points = np.random.randn(self.npoints, 3).astype(np.float32)
            labels = np.ones(self.npoints, dtype=np.int64) if labels is not None else None
        
        # Sample points
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(points), self.npoints, replace=True)
            
        point_set = points[choice, :]
        
        # Center and normalize
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
        if dist > 0:
            point_set = point_set / dist
        
        # Apply data augmentation
        if self.data_augmentation:
            # Random rotation around Y axis
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            point_set = point_set.dot(rotation_matrix)
            
            # Random jitter
            point_set += np.random.normal(0, 0.02, size=point_set.shape)
        
        # Convert to tensors
        point_set = torch.from_numpy(point_set)
        cls_label = torch.from_numpy(np.array([self.category_to_idx[category]]).astype(np.int64))
        
        if self.classification:
            return point_set, cls_label
        else:
            # For segmentation
            if labels is not None:
                seg_labels = labels[choice]
                seg_labels = torch.from_numpy(seg_labels)
                return point_set, seg_labels
            else:
                # No labels available, return dummy labels
                dummy_labels = torch.ones(self.npoints, dtype=torch.int64)
                return point_set, dummy_labels
    
    def __len__(self):
        return len(self.files)


def create_custom_shapenet_splits(root_dir, train_ratio=0.8):
    """
    Create train/test splits for custom ShapeNet data
    
    Args:
        root_dir: Root directory containing category folders
        train_ratio: Ratio of data to use for training
    """
    import json
    from sklearn.model_selection import train_test_split
    
    all_files = []
    categories = []
    
    # Scan all categories
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            txt_files = glob(os.path.join(category_path, "*.txt"))
            for txt_file in txt_files:
                rel_path = os.path.relpath(txt_file, root_dir)
                all_files.append(rel_path)
                categories.append(category)
    
    # Split data
    train_files, test_files = train_test_split(
        all_files, test_size=1-train_ratio, random_state=42, 
        stratify=categories
    )
    
    # Save splits
    splits = {
        'train': train_files,
        'test': test_files
    }
    
    os.makedirs(os.path.join(root_dir, 'train_test_split'), exist_ok=True)
    
    with open(os.path.join(root_dir, 'train_test_split', 'file_splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created splits: {len(train_files)} train, {len(test_files)} test files")
    return splits


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python custom_shapenet_dataset.py <data_root> [create_splits]")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == 'create_splits':
        create_custom_shapenet_splits(root_dir)
    
    # Test dataset loading
    try:
        dataset = CustomShapeNetDataset(root_dir, npoints=1000)
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        print(f"Number of segmentation classes: {dataset.num_seg_classes}")
        
        # Test loading first sample
        if len(dataset) > 0:
            points, labels = dataset[0]
            print(f"First sample - Points shape: {points.shape}, Labels shape: {labels.shape}")
            
    except Exception as e:
        print(f"Error testing dataset: {e}")
