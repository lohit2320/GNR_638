import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import numpy as np
import cv2
from ptflops import get_model_complexity_info
import gc

# --- Configuration ---
DATA_DIR = '../data/train_data'
NUM_CLASSES = 30
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = ['resnet50', 'densenet121', 'efficientnet_b0']
EPOCHS = 30  
RANDOM_SEED = 42


def apply_gauss_noise(tensor, sigma):
    return tensor + torch.randn_like(tensor) * sigma

def apply_motion_blur(img_tensor, kernel_size=9):
    device = img_tensor.device
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(img_np, -1, kernel)
    return torch.from_numpy(blurred).permute(2, 0, 1).to(device)

def apply_brightness_shift(tensor, value=0.3):
    return torch.clamp(tensor + value, 0, 1)

def run_robustness_task():
    # 1. Data Loading
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluation transform 
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    full_dataset_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_dataset_eval = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)
    
    train_size = int(0.8 * len(full_dataset_train))
    val_size = len(full_dataset_train) - train_size

    train_indices, val_indices = random_split(
        range(len(full_dataset_train)), [train_size, val_size], 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset_train, train_indices), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_subset_eval = torch.utils.data.Subset(full_dataset_eval, val_indices)
    
    final_results = {}

    for model_name in MODELS:
        print(f"\n{'#'*60}\n# Starting Task 4.4 for {model_name}\n{'#'*60}")
        
        # model initialisation
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # train
        print(f"--- Phase 1: Training on Clean Data (30 Epochs) ---")
        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} completed.")

        # evaluation on corrupted data
        print(f"--- Phase 2: Evaluating Robustness (Evaluation-time Corruptions) ---")
        model.eval()
        
        corruptions = {
            'Clean': lambda x: x,
            'Gauss_0.05': lambda x: apply_gauss_noise(x, 0.05),
            'Gauss_0.1': lambda x: apply_gauss_noise(x, 0.1),
            'Gauss_0.2': lambda x: apply_gauss_noise(x, 0.2),
            'MotionBlur': lambda x: apply_motion_blur(x, 9),
            'Brightness': lambda x: apply_brightness_shift(x, 0.3)
        }
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model_metrics = {}
        
        for c_name, c_func in corruptions.items():
            correct, total = 0, 0
            val_loader = DataLoader(val_subset_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    # Apply Corruption
                    corrupted_inputs = torch.stack([c_func(img) for img in inputs])
                    # Apply Normalization
                    final_inputs = torch.stack([normalize(img) for img in corrupted_inputs])
                    
                    outputs = model(final_inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            model_metrics[c_name] = acc
            print(f" >> {c_name:<12}: {acc:.2f}%")

        # Metrics
        clean_acc = model_metrics['Clean']
        processed_results = {}
        for c_name, acc in model_metrics.items():
            processed_results[c_name] = {
                'Accuracy': acc,
                'CorruptionError': 1.0 - (acc / 100.0),
                'RelativeRobustness': acc / clean_acc if clean_acc > 0 else 0
            }
            
        final_results[model_name] = processed_results
        
        # Memory Cleanup
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # Save Results
    with open('robustness_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    print("\nTask 4.4 Complete. Results saved to robustness_results.json")

if __name__ == '__main__':
    run_robustness_task()