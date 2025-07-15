#!/usr/bin/env python3
"""
NVIDIA Jetson Optimization Script for PointNet
Sets up environment and optimizations for running PointNet on Jetson devices
"""

import os
import subprocess
import sys
import json
import torch
import numpy as np


def check_jetson_environment():
    """Check if running on Jetson and get device info"""
    try:
        # Check for Jetson-specific files
        jetson_files = [
            '/etc/nv_tegra_release',
            '/sys/module/tegra_fuse/parameters/tegra_chip_id'
        ]
        
        is_jetson = any(os.path.exists(f) for f in jetson_files)
        
        if is_jetson:
            print("✓ Running on NVIDIA Jetson device")
            
            # Get Jetson model info
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip()
                print(f"  Model: {model}")
            except:
                pass
            
            # Check available memory
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            mem_total = line.split()[1]
                            print(f"  Total Memory: {int(mem_total)//1024} MB")
                            break
            except:
                pass
                
        else:
            print("⚠ Not running on Jetson device")
            
        return is_jetson
    except Exception as e:
        print(f"Error checking Jetson environment: {e}")
        return False


def optimize_cuda_settings():
    """Optimize CUDA settings for Jetson"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("Optimizing CUDA settings...")
    
    # Set memory fraction for Jetson
    torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Check GPU memory
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    print(f"  GPU: {gpu_name}")
    print(f"  Total GPU Memory: {total_memory // (1024**2)} MB")
    print(f"  Memory fraction set to: 80%")


def set_jetson_performance_mode():
    """Set Jetson to maximum performance mode"""
    try:
        # Try to set maximum performance mode
        result = subprocess.run(['sudo', 'nvpmodel', '-m', '0'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Set Jetson to maximum performance mode")
        else:
            print("⚠ Could not set performance mode (may need sudo)")
    except FileNotFoundError:
        print("⚠ nvpmodel not found (not on Jetson?)")
    except Exception as e:
        print(f"⚠ Error setting performance mode: {e}")


def set_jetson_fan_profile():
    """Set fan to maximum cooling"""
    try:
        # Try to set fan to maximum
        result = subprocess.run(['sudo', 'jetson_clocks'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Set Jetson clocks to maximum")
        else:
            print("⚠ Could not set jetson_clocks (may need sudo)")
    except FileNotFoundError:
        print("⚠ jetson_clocks not found (not on Jetson?)")
    except Exception as e:
        print(f"⚠ Error setting clocks: {e}")


def optimize_pytorch_settings():
    """Optimize PyTorch settings for Jetson"""
    print("Optimizing PyTorch settings...")
    
    # Set number of threads
    num_cores = os.cpu_count()
    torch.set_num_threads(min(num_cores, 4))  # Limit to 4 threads max
    
    # Set inter-op parallelism
    torch.set_num_interop_threads(2)
    
    print(f"  Set PyTorch threads: {torch.get_num_threads()}")
    print(f"  Set inter-op threads: {torch.get_num_interop_threads()}")


def create_jetson_config():
    """Create configuration file for Jetson optimizations"""
    config = {
        'batch_size': 16,  # Smaller batch size for Jetson
        'num_workers': 2,  # Fewer workers
        'npoints': 2500,   # Standard number of points
        'memory_fraction': 0.8,
        'enable_amp': True,  # Enable Automatic Mixed Precision
        'pin_memory': True,
        'non_blocking': True,
        'persistent_workers': True
    }
    
    config_path = os.path.join(os.path.dirname(__file__), 'jetson_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created Jetson configuration: {config_path}")
    return config


def install_jetson_dependencies():
    """Install dependencies optimized for Jetson"""
    print("Installing Jetson-optimized dependencies...")
    
    dependencies = [
        'torch',
        'torchvision', 
        'torchaudio',
        'numpy',
        'plyfile',
        'tqdm',
        'scikit-learn'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} already installed")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep])


def benchmark_pointnet():
    """Run a simple benchmark to test PointNet performance"""
    print("\nRunning PointNet benchmark...")
    
    try:
        # Import PointNet model
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from pointnet.model import PointNetDenseCls
        
        # Create model
        model = PointNetDenseCls(k=4)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        
        # Create dummy input
        batch_size = 8
        npoints = 2500
        dummy_input = torch.randn(batch_size, 3, npoints)
        
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        # Warm-up
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        num_iterations = 20
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output, _, _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = batch_size / avg_time
        
        print(f"✓ Benchmark completed:")
        print(f"  Average inference time: {avg_time:.4f} seconds")
        print(f"  Throughput: {fps:.2f} samples/second")
        print(f"  Points per second: {fps * npoints:.0f}")
        
    except Exception as e:
        print(f"⚠ Benchmark failed: {e}")


def main():
    print("=== NVIDIA Jetson PointNet Optimization ===\n")
    
    # Check environment
    is_jetson = check_jetson_environment()
    
    # Install dependencies
    install_jetson_dependencies()
    
    # Optimize settings
    optimize_pytorch_settings()
    optimize_cuda_settings()
    
    # Create configuration
    config = create_jetson_config()
    
    # Jetson-specific optimizations
    if is_jetson:
        set_jetson_performance_mode()
        set_jetson_fan_profile()
    
    # Run benchmark
    benchmark_pointnet()
    
    print("\n=== Optimization Complete ===")
    print("You can now run PointNet with optimized settings:")
    print("  - Use --jetson flag in training/inference scripts")
    print("  - Configuration saved to jetson_config.json")
    print("  - Recommended batch size: 16 or lower")
    print("  - Monitor temperature during training")


if __name__ == '__main__':
    main()
