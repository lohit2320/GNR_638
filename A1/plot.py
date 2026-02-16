import matplotlib.pyplot as plt
import csv

def main():
    epochs = []
    losses = []
    train_accs = []
    test_accs = []

    try:
        with open("training_logs.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                losses.append(float(row['train_loss']))
                train_accs.append(float(row['train_acc']))
                test_accs.append(float(row['test_acc']))
    except FileNotFoundError:
        print("Error: 'training_logs.csv' not found. Run final_submission.py first.")
        return

    # 1. Plot Loss
    plt.figure()
    plt.plot(epochs, losses, marker='o', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_loss.png')
    print("Saved plot_loss.png")

    # 2. Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accs, marker='o', label='Train Accuracy')
    plt.plot(epochs, test_accs, marker='s', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_accuracy.png')
    print("Saved plot_accuracy.png")

if __name__ == "__main__":
    main()