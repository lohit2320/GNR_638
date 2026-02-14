import os
import time
import my_backend
from src.utilities.dataloader import DataLoader

def get_argmax(flat_probs, num_classes):
    """Manual argmax for Python lists (to avoid NumPy )"""
    predictions = []
    for i in range(0, len(flat_probs), num_classes):
        chunk = flat_probs[i : i + num_classes]
        max_val = max(chunk)
        predictions.append(chunk.index(max_val))
    return predictions

def train_and_eval(dataset_name):
    DATA_PATH = os.path.join("dataset", dataset_name)
    if not os.path.exists(DATA_PATH):
        print(f"Path {DATA_PATH} not found!")
        return

    # A. Data Loading & Timing 
    loader = DataLoader(DATA_PATH, batch_size=32)
    print(f"\n--- DATASET: {dataset_name} ---")
    print(f"Dataset Loading Time (Disk Scan): {loader.disk_scan_time:.4f}s")
    
    num_classes = loader.num_classes
    
    # B. Model Setup (Conv, Activation, Pool, FC [cite: 42, 43, 44, 45, 46])
    # Architecture: 32x32x3 -> Conv3x3(8) -> 30x30x8 -> Pool2x2 -> 15x15x8 (1800) -> FC(num_classes)
    conv = my_backend.Conv2d(3, 8, 3, 1, 0)
    relu = my_backend.ReLU()
    pool = my_backend.MaxPool(2, 2)
    fc   = my_backend.FullyConnected(1800, num_classes)
    loss_fn = my_backend.SoftmaxClassifier()
    
    LR = 0.01

    # C. Training Loop [cite: 103]
    print(f"Training on {num_classes} classes...")
    for epoch in range(3):
        start_epoch = time.time()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for b_idx, (batch_x, batch_y) in enumerate(loader.get_batch_generator()):
            # Forward [cite: 15]
            t_in = my_backend.Tensor(batch_x, [len(batch_y), 3, 32, 32])
            x = conv.forward(t_in)
            x = relu.forward(x)
            x = pool.forward(x)
            x = fc.forward(x)
            
            # Prediction for Accuracy Metric
            probs = loss_fn.predict(x)
            preds = get_argmax(probs.to_list(), num_classes)
            for p, g in zip(preds, batch_y):
                if p == g: correct += 1
            total_samples += len(batch_y)
            
            # Backward [cite: 15, 25]
            loss, grad = loss_fn.backprop(batch_y)
            grad = fc.backprop(grad, LR)
            grad = pool.backprop(grad, LR)
            grad = relu.backprop(grad, LR)
            grad = conv.backprop(grad, LR)
            
            total_loss += loss
            
        avg_loss = total_loss / (total_samples // 32)
        accuracy = (correct / total_samples) * 100
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Time: {time.time()-start_epoch:.2f}s")

if __name__ == "__main__":
    # Process both datasets as provided in your screenshot
    for ds in ["data_1", "data_2"]:
        train_and_eval(ds)
        