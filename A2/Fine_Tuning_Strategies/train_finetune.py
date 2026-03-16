import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import json
import numpy as np
from ptflops import get_model_complexity_info
import time

# --- Configuration ---
DATA_DIR = '../data/train_data' #give path to dataset
NUM_CLASSES = 30
EPOCHS = 30
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = ['resnet50', 'densenet121', 'efficientnet_b0']
STRATEGIES = ['linear_probe', 'last_block', 'full', 'selective_20']


def apply_finetuning_strategy(model, model_name, strategy):
    for param in model.parameters():
        param.requires_grad = False
        
    total_params = sum(p.numel() for p in model.parameters())
    
    if strategy == 'linear_probe':
        for param in model.get_classifier().parameters(): param.requires_grad = True
            
    elif strategy == 'full':
        for param in model.parameters(): param.requires_grad = True
            
    elif strategy == 'last_block':
        for param in model.get_classifier().parameters(): param.requires_grad = True
        if 'resnet' in model_name:
            for param in model.layer4.parameters(): param.requires_grad = True
        elif 'densenet' in model_name:
            for param in model.features.denseblock4.parameters(): param.requires_grad = True
        elif 'efficientnet' in model_name:
            for param in model.blocks[-1].parameters(): param.requires_grad = True
            
    elif strategy == 'selective_20':
        for param in model.get_classifier().parameters(): param.requires_grad = True
        target_unfrozen = 0.20 * total_params
        current_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        for name, param in reversed(list(model.named_parameters())):
            if 'classifier' in name or 'fc' in name: continue 
            if current_unfrozen < target_unfrozen:
                param.requires_grad = True
                current_unfrozen += param.numel()
            else:
                break
                
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_unfrozen = float((unfrozen_params / total_params) * 100) # Cast to float for JSON
    return percent_unfrozen

def run_experiments():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    results = {}
    
    for model_name in MODELS:
        results[model_name] = {}
        for strategy in STRATEGIES:
            print(f"\n{'='*50}")
            print(f"Training {model_name} | Strategy: {strategy}")
            print(f"{'='*50}")
            
            model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
            percent_unfrozen = apply_finetuning_strategy(model, model_name, strategy)
            
            macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
            print(f"[Training Phase] Efficiency -> Params: {params} | MACs: {macs}")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            
            history = {
                'train_loss': [], 'train_acc': [], 'val_acc': [], 
                'percent_unfrozen': percent_unfrozen, 'grad_norms': {}
            }
            
            for epoch in range(EPOCHS):
                # print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
                start_time = time.time()
                # Training
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                epoch_grad_norms = {}
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            base_name = name.split('.')[0] 
                            norm = p.grad.norm().item()
                            epoch_grad_norms[base_name] = epoch_grad_norms.get(base_name, []) + [norm]
                            
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                train_acc = float(100. * correct / total)
                train_loss = float(running_loss / len(train_loader))
                
                for k, v in epoch_grad_norms.items():
                    if k not in history['grad_norms']: history['grad_norms'][k] = []
                    history['grad_norms'][k].append(float(np.mean(v))) 
                
                # Validation
                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        
                val_acc = float(100. * val_correct / val_total)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)

                epoch_time = time.time() - start_time
                print(f"Epoch: {epoch+1}/{EPOCHS} Time: {epoch_time:.2f}s Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
                
            results[model_name][strategy] = history
            
    with open('finetune_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nData saved to finetune_results.json")

if __name__ == '__main__':
    run_experiments()