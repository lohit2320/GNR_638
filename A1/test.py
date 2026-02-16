import os
import time
import cv2
import my_backend

# ================= CONFIGURATION =================
DATASET_PATH = "testdata"  # give path to test data (which has depth = 1 subdirectories)
WEIGHTS_DIR = "weights" # give path to the weights directory saved by train.py
BATCH_SIZE = 32 
IMAGE_SIZE = 32
CHANNELS = 3
TRAIN_SPLIT = 0
# =================================================

def load_weights(conv1, conv2, fc):
    print(f"\n[Loading] Reading weights from {WEIGHTS_DIR}/...")
    try:
        if not os.path.exists(os.path.join(WEIGHTS_DIR, "fc.txt")):
            print("Error: Weights not found! Run train.py first.")
            return False
        
        conv1.load(os.path.join(WEIGHTS_DIR, "conv1.txt"))
        conv2.load(os.path.join(WEIGHTS_DIR, "conv2.txt"))
        fc.load(os.path.join(WEIGHTS_DIR, "fc.txt"))
        print("Weights loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False

class TestLoader:
    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found at {data_dir}")

        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg'))])
            idx = int(len(files) * TRAIN_SPLIT)
            selected = files[idx:] 
            
            for f in selected:
                self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))
        
        print(f"[TEST] Found {self.num_classes} classes.")
        print(f"[TEST] Loaded {len(self.samples)} images.")

    def get_batch(self):
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i : i + self.batch_size]
            flat_pixels = []
            labels = []
            valid_batch_size = 0
            
            for path, label in batch:
                img = cv2.imread(path)
                if img is None: continue
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                
               
                img = img.transpose(2, 0, 1) 
                flat_pixels.extend((img / 255.0).flatten().tolist())
                
                labels.append(label)
                valid_batch_size += 1
            
            if valid_batch_size > 0:
                yield flat_pixels, labels, valid_batch_size

def get_argmax(flat_probs, num_classes):
    preds = []
    for i in range(0, len(flat_probs), num_classes):
        chunk = flat_probs[i : i + num_classes]
        preds.append(chunk.index(max(chunk)))
    return preds

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist.")
        return

    
    test_loader = TestLoader(DATASET_PATH, BATCH_SIZE)
    NUM_CLASSES = test_loader.num_classes
    
    print(f"\n>>> Initializing C++ Model for {NUM_CLASSES} classes...")

    conv1 = my_backend.Conv2d(3, 6, 5, 1, 0, 0)
    relu1 = my_backend.ReLU()
    pool1 = my_backend.MaxPool(2, 2)
    
    conv2 = my_backend.Conv2d(6, 16, 3, 1, 0, 0)
    relu2 = my_backend.ReLU()
    pool2 = my_backend.MaxPool(2, 2)
    
    fc = my_backend.FullyConnected(576, NUM_CLASSES, 0)
    loss_fn = my_backend.SoftmaxClassifier()

    if not load_weights(conv1, conv2, fc):
        return

    # Evaluation
    correct = 0
    total = 0
    
    print("\n>>> Starting Evaluation...")
    start_time = time.time()
    
    for flat_pixels, labels, bs in test_loader.get_batch():
        t_in = my_backend.Tensor(flat_pixels, [bs, CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
        
        # Forward 
        x = conv1.forward(t_in)
        x = relu1.forward(x)
        x = pool1.forward(x)
        x = conv2.forward(x)
        x = relu2.forward(x)
        x = pool2.forward(x)
        x = fc.forward(x)
        
        probs = loss_fn.predict(x)
        
        predictions = get_argmax(probs.to_list(), NUM_CLASSES)
        for p, g in zip(predictions, labels):
            if p == g: correct += 1
        total += bs
        print(f"Tested {total} images...", end='\r')

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nFinal Test Accuracy: {acc:.2f}%")
    print(f"Time Taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()