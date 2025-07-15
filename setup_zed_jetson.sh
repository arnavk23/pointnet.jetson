# Setup script for ZED camera + PointNet on NVIDIA Jetson

set -e

echo "=== ZED Camera PointNet Setup ==="
echo "Setting up PointNet for ZED camera semantic segmentation on NVIDIA Jetson"
echo

# Check if we're on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "✓ Detected NVIDIA Jetson device"
    IS_JETSON=true
else
    echo "⚠ Not running on Jetson device"
    IS_JETSON=false
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install numpy tqdm scikit-learn matplotlib
pip3 install plyfile

# Install PointNet
echo "Installing PointNet..."
pip3 install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data/shapenet
mkdir -p data/zed_output
mkdir -p results

# Set up Jetson optimizations
if [ "$IS_JETSON" = true ]; then
    echo "Applying Jetson optimizations..."
    
    # Set performance mode
    sudo nvpmodel -m 0 2>/dev/null || echo "Could not set nvpmodel (may need manual setup)"
    
    # Set maximum clocks
    sudo jetson_clocks 2>/dev/null || echo "Could not set jetson_clocks (may need manual setup)"
    
    # Create swap file for additional memory
    if [ ! -f /swapfile ]; then
        echo "Creating swap file for additional memory..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
fi

# Run optimization script
echo "Running optimization script..."
python3 utils/jetson_optimize.py

# Create example data structure
echo "Creating example data structure..."
mkdir -p data/shapenet/03001627  # Chair category
cat > data/shapenet/03001627/example.txt << 'EOF'
# Example point cloud data format
# x y z label
0.1 0.2 0.3 1
0.4 0.5 0.6 2
0.7 0.8 0.9 1
EOF

# Create configuration files
echo "Creating configuration files..."

# Training config
cat > train_config.json << 'EOF'
{
  "batch_size": 16,
  "num_workers": 2,
  "learning_rate": 0.001,
  "epochs": 25,
  "npoints": 2500,
  "feature_transform": true,
  "jetson_optimized": true
}
EOF

# Inference config
cat > inference_config.json << 'EOF'
{
  "npoints": 2500,
  "device": "cuda",
  "jetson_optimized": true,
  "memory_fraction": 0.8
}
EOF