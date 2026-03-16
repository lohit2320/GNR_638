import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Load Data
with open('few_shot_results/few_shot_results.json', 'r') as f:
    data = json.load(f)

PLOT_DIR = 'few_shot_results/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

models = list(data.keys())
ratios = ['1.0', '0.2', '0.05']
ratio_labels = ['100%', '20%', '5%']

# 1. Plot Validation Accuracy vs. Data Ratio
plt.figure(figsize=(10, 6))
for m in models:
    accs = [data[m][r]['final_val_acc'] for r in ratios]
    plt.plot(ratio_labels, accs, marker='s', linewidth=2, label=m)

plt.title('Validation Accuracy across different Data Regimes')
plt.xlabel('Percentage of Training Data used')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'val_acc_vs_ratio.png'))
plt.show()

# 2. Calculate Delta (Relative Performance Drop) and Overfitting Gap
print(f"{'Model':<15} | {'Acc 100%':<10} | {'Acc 5%':<10} | {'Delta (%)':<10} | {'Overfit Gap (5%)'}")
print("-" * 75)

delta_values = []
gap_values = []

for m in models:
    acc100 = data[m]['1.0']['final_val_acc']
    acc5 = data[m]['0.05']['final_val_acc']
    
    # Formula: Delta = (Acc100 - Acc5) / Acc100
    delta = (acc100 - acc5) / acc100 * 100
    delta_values.append(delta)
    
    # Overfitting Gap = Train Acc - Val Acc at 5%
    gap = data[m]['0.05']['final_train_acc'] - data[m]['0.05']['final_val_acc']
    gap_values.append(gap)
    
    print(f"{m:<15} | {acc100:<10.2f} | {acc5:<10.2f} | {delta:<10.2f} | {gap:.2f}%")

# 3. Bar Chart for Delta and Gap
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].bar(models, delta_values, color=['skyblue', 'salmon', 'lightgreen'])
ax[0].set_title('Relative Performance Drop (Δ)\n(Lower is more data efficient)')
ax[0].set_ylabel('Percentage Drop (%)')

ax[1].bar(models, gap_values, color=['skyblue', 'salmon', 'lightgreen'])
ax[1].set_title('Overfitting Gap at 5% Data\n(Train Acc - Val Acc)')
ax[1].set_ylabel('Accuracy Difference (%)')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'few_shot_metrics.png'))
plt.show()