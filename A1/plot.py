import matplotlib.pyplot as plt

# Data from your terminal output
epochs = [1, 2, 3, 4, 5]
train_acc = [52.99, 75.53, 81.66, 86.36, 89.44]
train_loss = [0.0453, 0.0241, 0.0184, 0.0137, 0.0107]

# Create Figure
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Accuracy (Blue Line)
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Accuracy (%)', color=color)
ax1.plot(epochs, train_acc, marker='o', color=color, linewidth=2, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  

# Plot Loss (Red Line)
color = 'tab:red'
ax2.set_ylabel('Avg Training Loss', color=color)  
ax2.plot(epochs, train_loss, marker='x', linestyle='--', color=color, linewidth=2, label='Loss')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Progress: Accuracy vs Loss')
fig.tight_layout()  
plt.savefig('accuracy_plot.png', dpi=300)
print("Plot saved as accuracy_plot.png")