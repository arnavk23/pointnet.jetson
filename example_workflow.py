#!/usr/bin/env python3
"""
Example workflow for ZED camera + PointNet semantic segmentation
This script demonstrates the complete pipeline from data preparation to inference
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    else:
        print(result.stdout)
        return True


def create_sample_data():
    """Create sample data for demonstration"""
    print("\n=== Creating Sample Data ===")
    
    # Create sample ShapeNet data
    shapenet_dir = Path("data/shapenet")
    shapenet_dir.mkdir(parents=True, exist_ok=True)
    
    # Create chair category (03001627)
    chair_dir = shapenet_dir / "03001627"
    chair_dir.mkdir(exist_ok=True)
    
    # Generate sample chair point clouds
    for i in range(10):
        points = []
        
        # Generate chair-like point cloud
        # Seat
        for _ in range(500):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.1, 0.1)
            z = np.random.uniform(-0.5, 0.5)
            points.append([x, y, z, 1])  # label 1 for seat
        
        # Back
        for _ in range(400):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(0.1, 0.8)
            z = np.random.uniform(-0.1, 0.1)
            points.append([x, y, z, 2])  # label 2 for back
        
        # Legs
        for _ in range(300):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.8, -0.1)
            z = np.random.uniform(-0.5, 0.5)
            points.append([x, y, z, 3])  # label 3 for legs
        
        # Arms
        for _ in range(200):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.1, 0.6)
            z = np.random.uniform(-0.5, -0.3)
            points.append([x, y, z, 4])  # label 4 for arms
        
        # Save sample file
        points = np.array(points)
        sample_file = chair_dir / f"sample_chair_{i:03d}.txt"
        np.savetxt(sample_file, points, fmt='%.6f')
    
    print(f"Created {len(list(chair_dir.glob('*.txt')))} sample chair files")
    
    # Create sample ZED PLY file
    zed_dir = Path("data/zed_output")
    zed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample ZED point cloud (simplified PLY format)
    zed_points = []
    for _ in range(2500):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        zed_points.append([x, y, z])
    
    # Create simple PLY file
    ply_file = zed_dir / "zed_out.ply"
    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(zed_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in zed_points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"Created sample ZED PLY file: {ply_file}")
    
    return True


def main():
    print("=== ZED Camera + PointNet Example Workflow ===")
    print("This script demonstrates the complete pipeline:")
    print("1. Data preparation")
    print("2. Model training")
    print("3. ZED inference")
    print("4. Result analysis")
    
    # Check if we're in the right directory
    if not os.path.exists("pointnet/model.py"):
        print("Error: Please run this script from the pointnet.pytorch directory")
        return
    
    # Create sample data
    create_sample_data()
    
    # Step 1: Prepare data
    print("\n=== Step 1: Data Preparation ===")
    success = run_command([
        "python3", "utils/prepare_shapenet_data.py",
        "--dataset", "data/shapenet",
        "--create_splits",
        "--fix_invalid"
    ], "Preparing ShapeNet data")
    
    if not success:
        print("Data preparation failed!")
        return
    
    # Step 2: Run Jetson optimization
    print("\n=== Step 2: Jetson Optimization ===")
    success = run_command([
        "python3", "utils/jetson_optimize.py"
    ], "Optimizing for Jetson")
    
    if not success:
        print("Jetson optimization failed!")
    
    # Step 3: Train model
    print("\n=== Step 3: Model Training ===")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    success = run_command([
        "python3", "utils/train_custom_segmentation.py",
        "--dataset", "data/shapenet",
        "--class_choice", "03001627",  # Chair category
        "--nepoch", "5",  # Few epochs for demo
        "--batchSize", "8",  # Small batch for demo
        "--outf", "models/chair_demo",
        "--jetson"
    ], "Training chair segmentation model")
    
    if not success:
        print("Training failed!")
        return
    
    # Step 4: Run inference
    print("\n=== Step 4: ZED Inference ===")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    model_file = "models/chair_demo/seg_model_best.pth"
    if not os.path.exists(model_file):
        # Try to find any model file
        model_files = list(Path("models/chair_demo").glob("seg_model_*.pth"))
        if model_files:
            model_file = str(model_files[0])
        else:
            print("No trained model found!")
            return
    
    success = run_command([
        "python3", "utils/zed_inference.py",
        "--input", "data/zed_output/zed_out.ply",
        "--output", "results/demo_segmentation.txt",
        "--model", model_file,
        "--class_choice", "Chair",
        "--jetson"
    ], "Running ZED inference")
    
    if not success:
        print("Inference failed!")
        return
    
    # Step 5: Analyze results
    print("\n=== Step 5: Results Analysis ===")
    
    results_file = "results/demo_segmentation.txt"
    if os.path.exists(results_file):
        print(f"Segmentation results saved to: {results_file}")
        
        # Quick analysis
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            # Count non-comment lines
            data_lines = [line for line in lines if not line.strip().startswith('#')]
            
            # Extract labels
            labels = []
            for line in data_lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        labels.append(int(parts[3]))
            
            if labels:
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"Segmentation analysis:")
                print(f"  Total points: {len(labels)}")
                print(f"  Label distribution:")
                for label, count in zip(unique_labels, counts):
                    percentage = count / len(labels) * 100
                    print(f"    Label {label}: {count} points ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
    
    print("\n=== Workflow Complete ===")
    print("Files created:")
    print("├── data/")
    print("│   ├── shapenet/03001627/     # Sample chair data")
    print("│   └── zed_output/zed_out.ply # Sample ZED data")
    print("├── models/chair_demo/         # Trained model")
    print("└── results/demo_segmentation.txt # Segmentation results")
    print()
    print("Next steps:")
    print("1. Replace sample data with your real ShapeNet data")
    print("2. Replace zed_out.ply with your real ZED camera output")
    print("3. Train on larger dataset for better performance")
    print("4. Optimize batch sizes and parameters for your Jetson device")


if __name__ == '__main__':
    main()
