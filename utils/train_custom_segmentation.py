"""
Training script for custom ShapeNet format segmentation
Supports format: category_id/object_id.txt
"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from utils.custom_shapenet_dataset import CustomShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json


def create_category_mapping(root_dir):
    """Create category mapping for ShapeNet IDs"""
    # Common ShapeNet category mappings
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
    
    # Check which categories exist in the data
    existing_categories = {}
    for category_id in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_id)
        if os.path.isdir(category_path):
            category_name = shapenet_categories.get(category_id, category_id)
            existing_categories[category_id] = category_name
    
    return existing_categories


def save_model_info(model_path, category_mapping, num_classes, args):
    """Save model information for inference"""
    # Convert all values to native Python types
    safe_args = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in vars(args).items()}
    
    # Convert other NumPy types if needed
    safe_info = {
        'num_classes': int(num_classes),  # ensure it's a plain int
        'category_mapping': {str(k): str(v) for k, v in category_mapping.items()},
        'model_args': safe_args,
        'npoints': int(args.npoints)
    }
    
    info_path = model_path.replace('.pth', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(safe_info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg_custom', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--class_choice', type=str, default=None, help="class choice (ShapeNet ID)")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--npoints', type=int, default=2500, help='number of points')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--jetson', action='store_true', help='optimize for NVIDIA Jetson')
    parser.add_argument('--create_splits', action='store_true', help='create train/test splits')
    
    args = parser.parse_args()
    print(args)
    
    # Jetson optimizations
    if args.jetson:
        print("Applying NVIDIA Jetson optimizations...")
        args.batchSize = min(args.batchSize, 16)  # Reduce batch size for Jetson
        args.workers = min(args.workers, 2)  # Reduce workers
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
    
    # Create category mapping
    category_mapping = create_category_mapping(args.dataset)
    print(f"Found categories: {category_mapping}")
    
    # Create train/test splits if requested
    if args.create_splits:
        from utils.custom_shapenet_dataset import create_custom_shapenet_splits
        create_custom_shapenet_splits(args.dataset)
    
    # Set random seed
    args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    # Determine class choice
    class_choice = None
    if args.class_choice:
        class_choice = [args.class_choice]
    
    # Create datasets
    print("Loading training dataset...")
    dataset = CustomShapeNetDataset(
        root=args.dataset,
        npoints=args.npoints,
        split='train',
        classification=False,
        class_choice=class_choice,
        data_augmentation=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("Loading test dataset...")
    test_dataset = CustomShapeNetDataset(
        root=args.dataset,
        npoints=args.npoints,
        split='test',
        classification=False,
        class_choice=class_choice,
        data_augmentation=False
    )
    
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training samples: {len(dataset)}, Test samples: {len(test_dataset)}")
    
    num_classes = dataset.num_seg_classes
    print(f'Number of segmentation classes: {num_classes}')
    
    # Create output directory
    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    
    # Initialize model
    classifier = PointNetDenseCls(k=num_classes, feature_transform=args.feature_transform)
    
    if args.model != '':
        print(f"Loading pretrained model: {args.model}")
        classifier.load_state_dict(torch.load(args.model))
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        print("Using CUDA")
    
    num_batch = len(dataset) // args.batchSize
    
    # Training loop
    best_test_acc = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'
    
    for epoch in range(args.nepoch):
        scheduler.step()
        
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, data in enumerate(dataloader):
            points, target = data
            points = points.transpose(2, 1).float
            
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            points = points.float()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1  # Convert to 0-indexed
            
            # Ensure targets are valid
            target = torch.clamp(target, 0, num_classes - 1)
            
            loss = F.nll_loss(pred, target)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            optimizer.step()
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            
            train_loss += loss.item()
            train_correct += correct.item()
            train_total += target.size(0)
            
            if i % 10 == 0:
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                    epoch, i, num_batch, loss.item(), correct.item() / target.size(0)))
        
        train_acc = train_correct / train_total
        train_loss /= len(dataloader)
        
        # Testing
        classifier.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for i, data in enumerate(testdataloader):
                points, target = data
                points = points.transpose(2, 1).float()
                
                if torch.cuda.is_available():
                    points, target = points.cuda(), target.cuda()
                
                points = points.float()
                pred, trans, trans_feat = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                
                # Ensure targets are valid
                target = torch.clamp(target, 0, num_classes - 1)
                
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                
                test_loss += loss.item()
                test_correct += correct.item()
                test_total += target.size(0)
        
        test_acc = test_correct / test_total
        test_loss /= len(testdataloader)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_path = os.path.join(args.outf, 'seg_model_best.pth')
            torch.save(classifier.state_dict(), model_path)
            save_model_info(model_path, category_mapping, num_classes, args)
            print(f'New best model saved: {model_path} (accuracy: {test_acc:.4f})')
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            model_path = os.path.join(args.outf, f'seg_model_{epoch}.pth')
            torch.save(classifier.state_dict(), model_path)
            save_model_info(model_path, category_mapping, num_classes, args)
    
    print(f'Training completed. Best test accuracy: {best_test_acc:.4f}')


if __name__ == '__main__':
    main()
