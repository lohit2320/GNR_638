import json
import matplotlib.pyplot as plt
import numpy as np

def visualize():
    with open('robustness_results.json', 'r') as f:
        data = json.load(f)
    
    models = list(data.keys())
    c_types = ['Clean', 'Gauss_0.05', 'Gauss_0.1', 'Gauss_0.2', 'MotionBlur', 'Brightness']
    
    plt.figure(figsize=(10, 6))
    for m in models:
        accs = [data[m][c]['Accuracy'] for c in c_types]
        plt.plot(c_types, accs, marker='o', label=m, linewidth=2)
    plt.title('Task 4.4: Accuracy under Corruptions (Trained on Clean Data)')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('robustness_accuracy.png')

    plt.figure(figsize=(10, 6))
    x = np.arange(len(c_types[1:])) 
    width = 0.2
    for i, m in enumerate(models):
        rel_vals = [data[m][c]['RelativeRobustness'] for c in c_types[1:]]
        plt.bar(x + i*width, rel_vals, width, label=m)
    plt.xticks(x + width, c_types[1:])
    plt.title('Relative Robustness Comparison')
    plt.ylabel('Ratio (Retained / Clean)')
    plt.legend()
    plt.savefig('robustness_relative.png')
    
    plt.show()

if __name__ == '__main__':
    visualize()