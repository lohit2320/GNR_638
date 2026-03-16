import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

# Load data
with open('finetune_results.json', 'r') as f:
    results = json.load(f)
MODELS = list(results.keys())
STRATEGIES = list(results[MODELS[0]].keys())

def plot_acc_vs_unfrozen():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Validation Accuracy vs Percentage of Unfrozen Parameters')
    
    for idx, model_name in enumerate(MODELS):
        unfrozen_pcts = [results[model_name][s]['percent_unfrozen'] for s in STRATEGIES]
        
        val_accs = [max(results[model_name][s]['val_acc']) for s in STRATEGIES] 
        sorted_indices = np.argsort(unfrozen_pcts)
        unfrozen_pcts = np.array(unfrozen_pcts)[sorted_indices]
        val_accs = np.array(val_accs)[sorted_indices]
        sorted_strategies = np.array(STRATEGIES)[sorted_indices]
        
        axes[idx].plot(unfrozen_pcts, val_accs, marker='o', linestyle='-', linewidth=2)
        for i, txt in enumerate(sorted_strategies):
            axes[idx].annotate(txt, (unfrozen_pcts[i], val_accs[i]), 
                            textcoords="offset points", xytext=(0,10), ha='center')
        
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel('% Unfrozen Parameters')
        axes[idx].set_ylabel('Max Validation Accuracy (%)')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig('acc_vs_unfrozen.png')
    plt.show()


def plot_acc2_vs_unfrozen():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('training Accuracy vs Percentage of Unfrozen Parameters')
    
    for idx, model_name in enumerate(MODELS):
        unfrozen_pcts = [results[model_name][s]['percent_unfrozen'] for s in STRATEGIES]
        val_accs = [max(results[model_name][s]['train_acc']) for s in STRATEGIES] 

        sorted_indices = np.argsort(unfrozen_pcts)
        unfrozen_pcts = np.array(unfrozen_pcts)[sorted_indices]
        val_accs = np.array(val_accs)[sorted_indices]
        sorted_strategies = np.array(STRATEGIES)[sorted_indices]
        
        axes[idx].plot(unfrozen_pcts, val_accs, marker='o', linestyle='-', linewidth=2)
        for i, txt in enumerate(sorted_strategies):
            axes[idx].annotate(txt, (unfrozen_pcts[i], val_accs[i]), 
                            textcoords="offset points", xytext=(0,10), ha='center')
            
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel('% Unfrozen Parameters')
        axes[idx].set_ylabel('Max training Accuracy (%)') 
        axes[idx].grid(True, linestyle='--', alpha=0.7)
            
    plt.tight_layout()
    plt.savefig('trainacc_vs_unfrozen.png')
    plt.show()

def plot_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Convergence Stability: Training Loss vs Epochs')
    
    for idx, model_name in enumerate(MODELS):
        for strategy in STRATEGIES:
            loss = results[model_name][strategy]['train_loss']
            axes[idx].plot(range(1, len(loss)+1), loss, label=strategy, linewidth=2)
            
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Training Loss')
        axes[idx].legend()
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig('convergence_stability.png')
    plt.show()

def plot_gradient_norms():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Average Gradient Norms Across Network Layers (Full Fine-Tuning)')
    
    for idx, model_name in enumerate(MODELS):
        grad_data = results[model_name]['full']['grad_norms']
        
        layers = list(grad_data.keys())
        avg_norms = [np.mean(grad_data[layer]) for layer in layers]
        
        axes[idx].bar(layers, avg_norms, color='skyblue')
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel('Layer Block')
        axes[idx].set_ylabel('Average Gradient Norm')
        axes[idx].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig('gradient_norms.png')
    plt.show()

plot_acc_vs_unfrozen()
plot_acc2_vs_unfrozen()
plot_convergence()
plot_gradient_norms()