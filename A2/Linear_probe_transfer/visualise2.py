import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')           
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report


MODELS        = ['resnet50', 'densenet121', 'efficientnet_b0']
NUM_CLASSES   = 30
SAVE_DIR      = 'saved_models'
PLOT_DIR      = os.path.join(SAVE_DIR, 'plots')
DATA_DIR      = '../data/train_data'
BATCH_SIZE    = 128
NUM_WORKERS   = 4
EMBED_SAMPLES = 30        
RANDOM_SEED   = 42

os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE = {'resnet50': '#1f77b4', 'densenet121': '#ff7f0e', 'efficientnet_b0': '#2ca02c'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names  = full_dataset.classes          
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

rng = np.random.default_rng(RANDOM_SEED)
all_targets = np.array([full_dataset.targets[i] for i in val_dataset.indices])
embed_indices = []
for c in range(NUM_CLASSES):
    c_idx = np.where(all_targets == c)[0]
    chosen = rng.choice(c_idx, size=min(EMBED_SAMPLES, len(c_idx)), replace=False)
    embed_indices.extend(chosen.tolist())

embed_subset  = Subset(val_dataset, embed_indices)
embed_loader  = DataLoader(embed_subset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS)
embed_labels  = np.array([val_dataset[i][1] for i in embed_indices])


def load_metrics(model_name):
    path = os.path.join(SAVE_DIR, f"{model_name}_metrics.json")
    with open(path) as f:
        return json.load(f)

def load_model(model_name, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    weights_path = os.path.join(SAVE_DIR, f"{model_name}_linear_probe.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

def extract_features_and_preds(model, loader, device):
    """Returns (features [N, D], predictions [N], labels [N])."""
    features, preds, labels = [], [], []

    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device)
            feats  = model.forward_features(inputs)  
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
            elif feats.dim() == 3:          
                feats = feats[:, 0]         
            features.append(feats.cpu())

            logits = model(inputs)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(lbls)

    features = torch.cat(features).numpy()
    preds    = torch.cat(preds).numpy()
    labels   = torch.cat(labels).numpy()
    return features, preds, labels

def plot_loss_curves(all_metrics):
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for mn, m in all_metrics.items():
        epochs = range(1, len(m['train_losses']) + 1)
        ax.plot(epochs, m['train_losses'], label=mn, color=PALETTE[mn])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss – All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'loss_all_models.png'), dpi=150)
    plt.close(fig)

    for mn, m in all_metrics.items():
        epochs = range(1, len(m['train_losses']) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, m['train_losses'], color=PALETTE[mn])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'Training Loss – {mn}')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'loss_{mn}.png'), dpi=150)
        plt.close(fig)

    print("  [✓] Loss curves saved.")


def plot_accuracy_curves(all_metrics):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mn, m in all_metrics.items():
        epochs = range(1, len(m['val_accuracies']) + 1)
        ax.plot(epochs, m['val_accuracies'], label=mn, color=PALETTE[mn])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Validation Accuracy – All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'val_accuracy_all_models.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mn, m in all_metrics.items():
        epochs = range(1, len(m['train_accuracies']) + 1)
        ax.plot(epochs, m['train_accuracies'], label=mn, color=PALETTE[mn])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Accuracy (%)')
    ax.set_title('Train Accuracy – All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'train_accuracy_all_models.png'), dpi=150)
    plt.close(fig)

    print("  [✓] Accuracy curves saved.")

def plot_confusion_matrix(model_name, preds, labels):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax, annot_kws={"size": 6})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    ax.set_title(f'Confusion Matrix – {model_name}', fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'confusion_matrix_{model_name}.png'), dpi=150)
    plt.close(fig)

    report_path = os.path.join(PLOT_DIR, f'classification_report_{model_name}.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(labels, preds, target_names=class_names))

    print(f"  [✓] Confusion matrix saved for {model_name}.")

from matplotlib.colors import ListedColormap

def _scatter_plot(proj, labels, title, save_path):

    colors = sns.color_palette("husl", 30) 
    custom_cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 9)) 

    sc = ax.scatter(proj[:, 0], proj[:, 1],
                    c=labels, 
                    cmap=custom_cmap, 
                    alpha=0.8, 
                    s=22, 
                    linewidths=0,
                    vmin=0, 
                    vmax=29)
    
    cbar = plt.colorbar(sc, ax=ax, ticks=range(30))
    cbar.ax.set_yticklabels(class_names, fontsize=7) 
    
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3) 
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_embeddings(model_name, features, labels):
    import umap.umap_ as umap

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    proj = pca.fit_transform(features)

    _scatter_plot(
        proj,
        labels,
        f'PCA – {model_name} (var: {pca.explained_variance_ratio_.sum()*100:.1f}%)',
        os.path.join(PLOT_DIR, f'pca_{model_name}.png')
    )
    print(f"  [✓] PCA saved for {model_name}.")

    n_components_pre = min(50, features.shape[1])
    pca_pre = PCA(n_components=n_components_pre, random_state=RANDOM_SEED)
    feats_pre = pca_pre.fit_transform(features)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("Loading training metrics …")
    all_metrics = {mn: load_metrics(mn) for mn in MODELS}

    print("\n[1] Plotting loss curves …")
    plot_loss_curves(all_metrics)

    print("\n[2] Plotting accuracy curves …")
    plot_accuracy_curves(all_metrics)

    for model_name in MODELS:
        print(f"\n{'─'*55}")
        print(f"  Processing: {model_name}")
        print(f"{'─'*55}")

        model = load_model(model_name, device)

        print("  [3] Building confusion matrix …")
        _, preds_val, labels_val = extract_features_and_preds(model, val_loader, device)
        plot_confusion_matrix(model_name, preds_val, labels_val)

        print("  [4] Extracting embeddings …")
        feats_embed, _, _ = extract_features_and_preds(model, embed_loader, device)
        plot_embeddings(model_name, feats_embed, embed_labels)

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"All plots saved to:  {os.path.abspath(PLOT_DIR)}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
