import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


results = torch.load('layer_probe_results.pt',weights_only=False)

MODELS = list(results.keys())
LAYERS = ['Early', 'Middle', 'Final']

def plot_accuracy_vs_depth():
    plt.figure(figsize=(10, 6))
    
    for model_name in MODELS:
        accs = [results[model_name][layer]['val_acc'] for layer in LAYERS]
        plt.plot(LAYERS, accs, marker='o', linewidth=2, label=model_name)
        
    plt.title('Validation Accuracy vs Network Depth')
    plt.xlabel('Layer Depth')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('acc_vs_depth.png')
    plt.show()


def plot_feature_norms():
    x = np.arange(len(LAYERS))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model_name in enumerate(MODELS):
        norms = [results[model_name][layer]['avg_norm'] for layer in LAYERS]
        offset = (i - 1) * width 
        ax.bar(x + offset, norms, width, label=model_name)
        
    ax.set_title('Average Feature L2 Norms Across Network Depth')
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Average L2 Norm')
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('feature_norms.png')
    plt.show()

def plot_pca_visualizations():
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('PCA 2D Feature Projections Across Models and Depths', fontsize=16)
    
    for row_idx, model_name in enumerate(MODELS):
        for col_idx, layer in enumerate(LAYERS):
            ax = axes[row_idx, col_idx]
            
            features = results[model_name][layer]['pca_features'].numpy()
            labels = results[model_name][layer]['pca_labels'].numpy()
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.7, s=20)
            
            if col_idx == 0:
                ax.set_ylabel(model_name, fontsize=14, fontweight='bold')
            if row_idx == 0:
                ax.set_title(f'{layer} Layer', fontsize=14)
                
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pca_visualizations.png')
    plt.show()

plot_accuracy_vs_depth()
plot_feature_norms()
plot_pca_visualizations()