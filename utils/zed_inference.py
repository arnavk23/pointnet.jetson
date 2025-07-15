#!/usr/bin/env python3
"""
ZED Camera Point Cloud Inference Script
Processes ZED camera generated PLY files and outputs semantic segmentation results
"""

import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.model import PointNetDenseCls
from plyfile import PlyData, PlyElement
import os
import json

def load_zed_ply(ply_path):
    """Load point cloud from ZED camera PLY file"""
    try:
        plydata = PlyData.read(ply_path)
        # Extract x, y, z coordinates
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        
        # Stack coordinates into point cloud
        points = np.vstack([x, y, z]).T
        
        # Check if color information exists
        color = None
        if 'red' in plydata['vertex'].dtype.names:
            r = plydata['vertex']['red']
            g = plydata['vertex']['green']
            b = plydata['vertex']['blue']
            color = np.vstack([r, g, b]).T
            
        return points, color
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None, None

def preprocess_points(points, npoints=2500):
    """Preprocess point cloud for PointNet input"""
    if points is None or len(points) == 0:
        return None
    
    # Remove NaN and infinite values
    valid_idx = np.isfinite(points).all(axis=1)
    points = points[valid_idx]
    
    if len(points) == 0:
        print("No valid points after filtering")
        return None
    
    # Sample or duplicate points to get exactly npoints
    if len(points) > npoints:
        choice = np.random.choice(len(points), npoints, replace=False)
    else:
        choice = np.random.choice(len(points), npoints, replace=True)
    
    point_set = points[choice, :]
    
    # Center the point cloud
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
    
    # Scale to unit sphere
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
    if dist > 0:
        point_set = point_set / dist
    
    return point_set.astype(np.float32)

def load_model(model_path, num_classes):
    """Load trained PointNet model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')
    classifier = PointNetDenseCls(k=num_classes)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    
    return classifier

def run_inference(classifier, points, device='cpu'):
    """Run inference on preprocessed points"""
    # Convert to tensor and add batch dimension
    point_tensor = torch.from_numpy(points).transpose(1, 0).contiguous()
    point_tensor = point_tensor.view(1, point_tensor.size()[0], point_tensor.size()[1])
    
    if device == 'cuda':
        point_tensor = point_tensor.cuda()
        classifier = classifier.cuda()
    
    # Run inference
    with torch.no_grad():
        pred, _, _ = classifier(point_tensor)
        pred_choice = pred.data.max(2)[1]
    
    return pred_choice.cpu().numpy()[0]

def save_segmentation_result(output_path, points, labels, class_names=None):
    """Save segmentation results to text file"""
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Point Cloud Segmentation Results\n")
        f.write("# Format: x y z label\n")
        if class_names:
            f.write("# Class mapping:\n")
            for i, name in enumerate(class_names):
                f.write(f"# {i}: {name}\n")
        f.write("\n")
        
        # Write points and labels
        for i in range(len(points)):
            x, y, z = points[i]
            label = labels[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {label}\n")
    
    print(f"Segmentation results saved to: {output_path}")

def get_class_names(class_choice):
    """Get class names for the segmentation classes"""
    # Define class mappings for common ShapeNet categories
    class_mappings = {
        'Chair': ['back', 'seat', 'leg', 'arm'],
        'Table': ['top', 'leg', 'support'],
        'Airplane': ['body', 'wing', 'tail', 'engine'],
        'Car': ['body', 'wheel', 'window', 'light'],
        'Lamp': ['base', 'shade', 'bulb', 'switch'],
        'Guitar': ['body', 'neck', 'head'],
        'Motorbike': ['body', 'wheel', 'handlebar', 'seat', 'engine', 'fuel_tank']
    }
    
    return class_mappings.get(class_choice, [f'part_{i}' for i in range(10)])

def main():
    parser = argparse.ArgumentParser(description='ZED Camera Point Cloud Semantic Segmentation')
    parser.add_argument('--input', type=str, required=True, help='Input ZED PLY file path')
    parser.add_argument('--output', type=str, required=True, help='Output segmentation TXT file path')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--class_choice', type=str, default='Chair', help='Object class for segmentation')
    parser.add_argument('--npoints', type=int, default=2500, help='Number of points to sample')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--jetson', action='store_true', help='Optimize for NVIDIA Jetson')
    
    args = parser.parse_args()
    
    # Jetson-specific optimizations
    if args.jetson:
        print("Applying NVIDIA Jetson optimizations...")
        if torch.cuda.is_available():
            args.device = 'cuda'
            # Set memory fraction for Jetson
            torch.cuda.set_per_process_memory_fraction(0.8)
        # Reduce batch size and enable half precision if needed
        torch.backends.cudnn.benchmark = True
    
    print(f"Loading ZED PLY file: {args.input}")
    points, color = load_zed_ply(args.input)
    
    if points is None:
        print("Failed to load point cloud")
        return
    
    print(f"Loaded {len(points)} points")
    
    # Preprocess points
    processed_points = preprocess_points(points, args.npoints)
    if processed_points is None:
        print("Failed to preprocess points")
        return
    
    print(f"Preprocessed to {len(processed_points)} points")
    
    # Load segmentation class info
    num_seg_classes_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc/num_seg_classes.txt')
    num_classes = 4  # Default for Chair
    
    if os.path.exists(num_seg_classes_file):
        with open(num_seg_classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == args.class_choice:
                    num_classes = int(parts[1])
                    break
    
    print(f"Using {num_classes} segmentation classes for {args.class_choice}")
    
    # Load model
    print(f"Loading model: {args.model}")
    classifier = load_model(args.model, num_classes)
    
    # Run inference
    print("Running segmentation inference...")
    pred_labels = run_inference(classifier, processed_points, args.device)
    
    # Add 1 to labels to match ShapeNet format (1-indexed)
    pred_labels = pred_labels + 1
    
    # Get class names
    class_names = get_class_names(args.class_choice)
    
    # Save results
    save_segmentation_result(args.output, processed_points, pred_labels, class_names)
    
    print(f"Segmentation complete! Results saved to: {args.output}")
    print(f"Label distribution: {np.bincount(pred_labels)}")

if __name__ == '__main__':
    main()
