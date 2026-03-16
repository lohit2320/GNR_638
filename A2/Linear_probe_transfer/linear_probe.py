import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
from ptflops import get_model_complexity_info
import time

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = '../data/train_data' # give the path to dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42) 
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4,pin_memory=True)


def setup_linear_probe(model_name, num_classes=30):
    print(f"--- Setting up {model_name} for Linear Probing ---")
    
    model = timm.create_model(model_name, pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    model.reset_classifier(num_classes)
    
    macs, params = get_model_complexity_info(model, (3, 224, 224), 
                                             as_strings=True, 
                                             print_per_layer_stat=False, 
                                             verbose=False)
    print(f"Model: {model_name}")
    print(f"Total Parameters: {params}")
    print(f"MACs: {macs}")
    
    return model


def train_linear_probe(model, train_loader, val_loader, model_name, save_dir, epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # loss
    criterion = nn.CrossEntropyLoss()
    
    # optimiser
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    train_losses, train_accuracies, val_accuracies = [], [], []

    print("Starting Training...")
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        # Validation 
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # Save model weights
    weights_path = os.path.join(save_dir, f"{model_name}_linear_probe.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights -> {weights_path}")

    #saving metrics in json 
    metrics = {
        'model_name': model_name,
        'epochs': epochs,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics  -> {metrics_path}")

    return train_losses, train_accuracies, val_accuracies


models_to_test = ['resnet50', 'densenet121', 'efficientnet_b0']
# Directory to store saved weights and metrics
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

all_results = {}

for model_name in models_to_test:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    model = setup_linear_probe(model_name, num_classes=30)
    train_losses, train_accs, val_accs = train_linear_probe(
        model, train_loader, val_loader,
        model_name=model_name,
        save_dir=save_dir,
        epochs=30
    )
    all_results[model_name] = {
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
    }
    print(f"Best Val Acc [{model_name}]: {max(val_accs):.2f}% (epoch {val_accs.index(max(val_accs))+1})")

print("\nAll models trained and saved.")