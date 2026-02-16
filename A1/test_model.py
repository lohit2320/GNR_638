import os
import time
import random
import cv2
import my_backend  # Your compiled C++ module

# ================= CONFIGURATION =================
DATASET_PATH = "dataset/data_1"  # Change this to your dataset folder
WEIGHTS_DIR = "weights"
BATCH_SIZE = 32      # Small batch size to avoid segfaults with C++ backend
EPOCHS = 5
LEARNING_RATE = 0.005
TRAIN_SPLIT = 0.8
IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
# =================================================

def save_weights(conv1, conv2, fc):
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    print(f"\n[Saving] Writing weights to {WEIGHTS_DIR}/...")
    try:
        # These strings are passed to the C++ wrapper we wrote
        conv1.save(os.path.join(WEIGHTS_DIR, "conv1.txt"))
        conv2.save(os.path.join(WEIGHTS_DIR, "conv2.txt"))
        fc.save(os.path.join(WEIGHTS_DIR, "fc.txt"))
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

class DataLoader:
    def __init__(self, data_dir, batch_size, split='train'):
        self.batch_size = batch_size
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg'))])
            random.seed(42) 
            random.shuffle(files)
            
            # 80/20 Split
            idx = int(len(files) * TRAIN_SPLIT)
            selected = files[:idx] if split == 'train' else files[idx:]
            
            for f in selected:
                self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} images.")

    def get_batch(self):
        random.shuffle(self.samples)
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i : i + self.batch_size]
            if not batch: continue
            
            flat_pixels = []
            labels = []
            valid_batch_size = 0
            
            for path, label in batch:
                img = cv2.imread(path)
                if img is None: continue
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                
                # Normalize and Flatten (Channel-First for your C++ Tensor)
                # Structure: RRR...GGG...BBB...
                for c in range(CHANNELS):
                    for h in range(IMAGE_SIZE):
                        for w in range(IMAGE_SIZE):
                            flat_pixels.append(float(img[h, w, c] / 255.0))
                
                labels.append(label)
                valid_batch_size += 1
            
            if valid_batch_size > 0:
                yield flat_pixels, labels, valid_batch_size

def get_argmax(flat_probs, num_classes):
    preds = []
    # flat_probs is a 1D list of [batch * 10]
    for i in range(0, len(flat_probs), num_classes):
        chunk = flat_probs[i : i + num_classes]
        preds.append(chunk.index(max(chunk)))
    return preds

def main():
    print(">>> Initializing C++ Model...")
    # 1. Architecture
    # Conv1: 3 in -> 6 out, 5x5 kernel
    conv1 = my_backend.Conv2d(3, 6, 5, 1, 0, 42) 
    relu1 = my_backend.ReLU()
    pool1 = my_backend.MaxPool(2, 2)
    
    # Conv2: 6 in -> 16 out, 3x3 kernel
    conv2 = my_backend.Conv2d(6, 16, 3, 1, 0, 43)
    relu2 = my_backend.ReLU()
    pool2 = my_backend.MaxPool(2, 2)
    
    # FC: 576 input (16 * 6 * 6) -> 10 output
    fc = my_backend.FullyConnected(576, NUM_CLASSES, 44)
    loss_fn = my_backend.SoftmaxClassifier()

    # 2. Data
    train_loader = DataLoader(DATASET_PATH, BATCH_SIZE, 'train')
    
    # 3. Training Loop
    print(f"\n>>> Starting Training ({EPOCHS} Epochs)")
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for flat_pixels, labels, bs in train_loader.get_batch():
            # Create C++ Tensor
            # Shape: [Batch, Channels, Height, Width]
            t_in = my_backend.Tensor(flat_pixels, [bs, CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
            
            # --- Forward ---
            x = conv1.forward(t_in)
            x = relu1.forward(x)
            x = pool1.forward(x)
            
            x = conv2.forward(x)
            x = relu2.forward(x)
            x = pool2.forward(x)
            
            x = fc.forward(x)
            
            # --- Loss & Backprop ---
            probs = loss_fn.predict(x)
            loss, grad = loss_fn.backprop(labels)
            
            # --- Backward ---
            grad = fc.backprop(grad, LEARNING_RATE)
            grad = pool2.backprop(grad, LEARNING_RATE)
            grad = relu2.backprop(grad, LEARNING_RATE)
            grad = conv2.backprop(grad, LEARNING_RATE)
            grad = pool1.backprop(grad, LEARNING_RATE)
            grad = relu1.backprop(grad, LEARNING_RATE)
            grad = conv1.backprop(grad, LEARNING_RATE)
            
            # --- Metrics ---
            total_loss += loss
            predictions = get_argmax(probs.to_list(), NUM_CLASSES)
            for p, g in zip(predictions, labels):
                if p == g: correct += 1
            total_samples += bs
            
            print(f"Ep {epoch+1} | Loss: {loss:.4f} | Acc: {(correct/total_samples)*100:.1f}%", end='\r')

        epoch_acc = (correct / total_samples) * 100
        print(f"\nEpoch {epoch+1} Done | Avg Loss: {total_loss/total_samples:.4f} | Acc: {epoch_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

    # 4. Save
    save_weights(conv1, conv2, fc)

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
    else:
        main()