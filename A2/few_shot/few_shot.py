import os
import json
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import timm
import numpy as np
from ptflops import get_model_complexity_info

# --- Configuration ---
DATA_DIR = '../data/train_data'  # give data path 
NUM_CLASSES = 30
SAVE_DIR = 'few_shot_results'
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
BATCH_SIZE = 64  # Reduced to 64 to prevent Out-Of-Memory on DenseNet
MODELS_TO_TEST = ['resnet50', 'densenet121', 'efficientnet_b0']
RATIOS = [1.0, 0.2, 0.05]

def get_stratified_subset(train_subset_obj, ratio, seed):
    """Correctly samples from a Subset object while maintaining class balance."""
    if ratio == 1.0:
        return train_subset_obj
    
    root_dataset = train_subset_obj.dataset
    split_indices = np.array(train_subset_obj.indices)
    
    # Get targets for ONLY the indices in the training split
    full_targets = np.array(root_dataset.targets)
    split_targets = full_targets[split_indices]
    
    indices_to_keep = []
    np.random.seed(seed)
    
    for class_idx in range(len(root_dataset.classes)):
        class_mask = np.where(split_targets == class_idx)[0]
        actual_class_indices = split_indices[class_mask]
        
        if len(actual_class_indices) > 0:
            n_samples = max(1, int(len(actual_class_indices) * ratio))
            subset_choice = np.random.choice(actual_class_indices, n_samples, replace=False)
            indices_to_keep.extend(subset_choice.tolist())
    
    return Subset(root_dataset, indices_to_keep)

def train_one_scenario(model_name, ratio):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {model_name} | Data Ratio: {int(ratio*100)}%")
    print(f"{'='*60}")

    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_base, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    current_train_ds = get_stratified_subset(train_base, ratio, RANDOM_SEED)
    
    train_loader = DataLoader(current_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # model initialisatin
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    
    # Efficiency Metrics
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, 
                                             print_per_layer_stat=False, verbose=False)
    print(f"Model Stats -> Params: {params}, MACs: {macs}")

    # Training 
    epochs = 30 if ratio == 1.0 else 20  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                v_total += labels.size(0)
                v_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * v_correct / v_total
        
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {history['train_loss'][-1]:.4f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    # Saving results
    result_data = {
        'model_name': model_name,
        'ratio': ratio,
        'params': params,
        'macs': macs,
        'history': history,
        'final_val_acc': val_acc,
        'final_train_acc': train_acc
    }
    
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return result_data

if __name__ == '__main__':
    all_results = {}
    
    for m_name in MODELS_TO_TEST:
        all_results[m_name] = {}
        for r in RATIOS:
            res = train_one_scenario(m_name, r)
            all_results[m_name][str(r)] = res
            
            # Intermediate save in case of crash
            with open(os.path.join(SAVE_DIR, 'few_shot_results.json'), 'w') as f:
                json.dump(all_results, f, indent=4)

    print("\nAll experiments completed. Results saved to few_shot_results.json")