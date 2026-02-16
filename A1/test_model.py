import time
import random
import my_backend
import os
import cv2

# --- CONFIG ---
BATCH_SIZE = 2  # Keep it 2 as proven by debug script
EPOCHS = 3
LR = 0.005

class DataLoader:
    def __init__(self, data_dir, batch_size, mode='train'):
        self.batch_size = batch_size
        self.classes = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.image_paths = []
        
        # Limit images to ensure completion in < 2 hours
        limit = 3000 if mode == 'train' else 1000
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg'))])
            # Deterministic shuffle
            random.seed(42)
            random.shuffle(files)
            files = files[:limit]
            
            for f in files:
                self.image_paths.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))
        
        print(f"[{mode}] Loaded {len(self.image_paths)} images.")

    def get_batch(self):
        random.shuffle(self.image_paths)
        for i in range(0, len(self.image_paths), self.batch_size):
            batch = self.image_paths[i : i + self.batch_size]
            batch_x = []
            batch_y = []
            
            for path, label in batch:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None: continue
                img = cv2.resize(img, (32, 32))
                
                # Standardize inputs
                for c in range(3): 
                    for h in range(32):
                        for w in range(32):
                            batch_x.append(float(img[h, w, c] / 255.0))
                batch_y.append(label)
            
            if len(batch_x) != (len(batch_y) * 3 * 32 * 32): continue
            yield batch_x, batch_y

def get_argmax(flat_probs, num_classes):
    preds = []
    for i in range(0, len(flat_probs), num_classes):
        chunk = flat_probs[i : i + num_classes]
        preds.append(chunk.index(max(chunk)))
    return preds

def main():
    DATA_PATH = "dataset/data_1"
    if not os.path.exists(DATA_PATH): return

    print("--- INIT ---")
    train_loader = DataLoader(DATA_PATH, BATCH_SIZE, 'train')
    test_loader  = DataLoader(DATA_PATH, BATCH_SIZE, 'test')
    
    # SAFE ARCHITECTURE (5x5 -> 3x3)
    conv1 = my_backend.Conv2d(3, 6, 5, 1, 0)
    relu1 = my_backend.ReLU()
    pool1 = my_backend.MaxPool(2, 2)
    conv2 = my_backend.Conv2d(6, 16, 3, 1, 0)
    relu2 = my_backend.ReLU()
    pool2 = my_backend.MaxPool(2, 2)
    fc = my_backend.FullyConnected(576, 10)
    loss_fn = my_backend.SoftmaxClassifier()
    
    print("\n>>> STARTING TRAINING (Fixed Order)")
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss = 0
        train_correct = 0
        total_samples = 0
        
        # --- TRAIN ---
        for b_idx, (batch_x, batch_y) in enumerate(train_loader.get_batch()):
            try:
                bs = len(batch_y)
                t_in = my_backend.Tensor(batch_x, [bs, 3, 32, 32])
                
                # Forward
                x = conv1.forward(t_in)
                x = relu1.forward(x)
                x = pool1.forward(x)
                x = conv2.forward(x)
                x = relu2.forward(x)
                x = pool2.forward(x)
                x = fc.forward(x)
                
                # CRITICAL FIX: Predict FIRST to populate output_
                probs = loss_fn.predict(x)
                
                # NOW Backprop works because output_ is set
                loss, grad = loss_fn.backprop(batch_y)
                
                # Optimization
                grad = fc.backprop(grad, LR)
                grad = pool2.backprop(grad, LR)
                grad = relu2.backprop(grad, LR)
                grad = conv2.backprop(grad, LR)
                grad = pool1.backprop(grad, LR)
                grad = relu1.backprop(grad, LR)
                grad = conv1.backprop(grad, LR)
                
                # Stats
                train_loss += loss
                preds = get_argmax(probs.to_list(), 10)
                for p, g in zip(preds, batch_y):
                    if p == g: train_correct += 1
                total_samples += bs

                if b_idx % 100 == 0:
                    print(f"Ep {epoch+1} | Batch {b_idx} | Loss: {loss:.4f}", end='\r')
                    
                del t_in, x, grad, probs

            except Exception as e:
                print(f"Skipped batch: {e}")
                continue
        
        # --- TEST ---
        test_correct = 0
        test_total = 0
        for batch_x, batch_y in test_loader.get_batch():
            bs = len(batch_y)
            t_in = my_backend.Tensor(batch_x, [bs, 3, 32, 32])
            x = conv1.forward(t_in)
            x = relu1.forward(x)
            x = pool1.forward(x)
            x = conv2.forward(x)
            x = relu2.forward(x)
            x = pool2.forward(x)
            x = fc.forward(x)
            
            probs = loss_fn.predict(x)
            preds = get_argmax(probs.to_list(), 10)
            for p, g in zip(preds, batch_y):
                if p == g: test_correct += 1
            test_total += bs
            del t_in, x

        acc = (train_correct / total_samples) * 100 if total_samples > 0 else 0
        te_acc = (test_correct / test_total) * 100 if test_total > 0 else 0
        print(f"\nEpoch {epoch+1} | Loss: {train_loss/total_samples:.4f} | Train Acc: {acc:.2f}% | Test Acc: {te_acc:.2f}% | Time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()