import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import timm
from ptflops import get_model_complexity_info
import numpy as np
import time

# --- Configuration ---
DATA_DIR = '../data/train_data' #give path to dataset
EPOCHS = 30 
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = ['resnet50', 'densenet121', 'efficientnet_b0']
LAYER_INDICES = [0, 2, 4] 
LAYER_NAMES = ['Early', 'Middle', 'Final']

def get_pca_fixed_subset(dataset, samples_per_class=30, num_classes=30):
    """Creates a fixed subset with exactly 30 samples per class."""
    class_counts = {i: 0 for i in range(num_classes)}
    subset_indices = []
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if class_counts[label] < samples_per_class:
            subset_indices.append(idx)
            class_counts[label] += 1
        if all(count == samples_per_class for count in class_counts.values()):
            break
            
    return Subset(dataset, subset_indices)

def run_layer_probing():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,pin_memory=True)
    
    pca_subset = get_pca_fixed_subset(val_dataset, samples_per_class=30, num_classes=30)
    pca_loader = DataLoader(pca_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {}

    for model_name in MODELS:
        print(f"\n{'='*50}\nEvaluating {model_name}\n{'='*50}")
        results[model_name] = {}
        
        # Load backbone to extract multi-scale features
        backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=LAYER_INDICES).to(DEVICE)
        backbone.eval() 

        with torch.no_grad():
            dummy_out = backbone(torch.randn(1, 3, 224, 224).to(DEVICE))
            feature_dims = [out.shape[1] for out in dummy_out]
            
        for idx, (layer_idx, layer_name) in enumerate(zip(LAYER_INDICES, LAYER_NAMES)):
            print(f"\n--- Probing {layer_name} Layer (Index {layer_idx}, Features: {feature_dims[idx]}) ---")
            
            # 1. Setup Linear Classifier for this specific depth
            pool = nn.AdaptiveAvgPool2d(1).to(DEVICE)
            classifier = nn.Linear(feature_dims[idx], 30).to(DEVICE)
            
            optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Print MACs/Params for the feature extraction up to this layer
            macs, params = get_model_complexity_info(backbone, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
            print(f"[Efficiency] Backbone up to this layer -> Params: {params} | MACs: {macs}")
            
            # 2. Train the Linear Probe
            for epoch in range(EPOCHS):
                start_time = time.time()
                classifier.train()
                correct, total = 0, 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    with torch.no_grad():
                        # Extract features and select the specific layer's output
                        features = backbone(inputs)[idx]
                        pooled_features = pool(features).flatten(1)
                        
                    optimizer.zero_grad()
                    outputs = classifier(pooled_features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time: .2f}s Train Acc: {100.*correct/total:.2f}%")
                
            # 3. Evaluate & Calculate Feature Norms
            classifier.eval()
            val_correct, val_total = 0, 0
            feature_norms = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    features = backbone(inputs)[idx]
                    pooled_features = pool(features).flatten(1)
                    
                    # Track L2 norm of the features for this layer
                    norms = torch.norm(pooled_features, p=2, dim=1)
                    feature_norms.extend(norms.cpu().numpy())
                    
                    outputs = classifier(pooled_features)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
            val_acc = 100. * val_correct / val_total
            avg_norm = np.mean(feature_norms)
            print(f"> Final Val Acc: {val_acc:.2f}% | Avg Feature Norm: {avg_norm:.4f}")
            
            # 4. Extract features for the fixed PCA subset
            pca_features, pca_labels = [], []
            with torch.no_grad():
                for inputs, labels in pca_loader:
                    inputs = inputs.to(DEVICE)
                    features = backbone(inputs)[idx]
                    pooled_features = pool(features).flatten(1)
                    pca_features.append(pooled_features.cpu())
                    pca_labels.append(labels)
                    
            pca_features = torch.cat(pca_features, dim=0)
            pca_labels = torch.cat(pca_labels, dim=0)
            
            # Save results for this layer
            results[model_name][layer_name] = {
                'val_acc': val_acc,
                'avg_norm': avg_norm,
                'pca_features': pca_features,
                'pca_labels': pca_labels
            }
            
    # Save the aggregated results
    torch.save(results, 'layer_probe_results.pt')
    print("\nAll layer probing results saved to 'layer_probe_results.pt'")

if __name__ == '__main__':
    run_layer_probing()