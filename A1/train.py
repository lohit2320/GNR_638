import os
import time
import random
import cv2
import my_backend  

# ================= CONFIGURATION =================
DATASET_PATH = "dataset/data_2"    # give path of data_set to train (data_1 && data_2) (depth =1 subdirectories)
WEIGHTS_DIR = "weights"   
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.005
TRAIN_SPLIT = 1
IMAGE_SIZE = 32
CHANNELS = 3
# =================================================

def save_weights(conv1, conv2, fc):
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    print(f"\n[Saving] Writing weights to {WEIGHTS_DIR}/...")
    try:
        conv1.save(os.path.join(WEIGHTS_DIR, "conv1.txt"))
        conv2.save(os.path.join(WEIGHTS_DIR, "conv2.txt"))
        fc.save(os.path.join(WEIGHTS_DIR, "fc.txt"))
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

class DataLoader:
    def __init__(self, data_dir, batch_size, split='train'):
        self.batch_size = batch_size
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found.")
            
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg'))])
            random.seed(42) 
            random.shuffle(files)
            
            idx = int(len(files) * TRAIN_SPLIT)
            selected = files[:idx] if split == 'train' else files[idx:]
            
            for f in selected:
                self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))        
        
        print(f"[{split.upper()}] Found {self.num_classes} classes: {self.classes}")


    def get_batch(self,measure_time=False):

        total_disk_time = 0
        total_batches = 0
        random.shuffle(self.samples)
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i : i + self.batch_size]
            if not batch: continue
            
            flat_pixels = []
            labels = []
            valid_batch_size = 0
            for path, label in batch:
                start_disk = time.perf_counter()
                img = cv2.imread(path)
                end_disk = time.perf_counter()
                total_disk_time += (end_disk - start_disk)
                if img is None: continue
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = img.transpose(2, 0, 1)
                flat_pixels.extend((img / 255.0).flatten().tolist())
                labels.append(label)
                valid_batch_size += 1
            
            if valid_batch_size > 0:
                total_batches += 1
                yield flat_pixels, labels, valid_batch_size
            
        if measure_time:
            print(f"Total Disk Read Time: {total_disk_time:.4f} sec")

def get_argmax(flat_probs, num_classes):
    preds = []
    for i in range(0, len(flat_probs), num_classes):
        chunk = flat_probs[i : i + num_classes]
        preds.append(chunk.index(max(chunk)))
    return preds

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    train_loader = DataLoader(DATASET_PATH, BATCH_SIZE, 'train')
    NUM_CLASSES = train_loader.num_classes
    
    print(f"\n>>> Initializing C++ Model (Classes: {NUM_CLASSES})...")
    
    #  Architecture
    conv1 = my_backend.Conv2d(3, 6, 5, 1, 0, 42) 
    relu1 = my_backend.ReLU()
    pool1 = my_backend.MaxPool(2, 2)
    
    conv2 = my_backend.Conv2d(6, 16, 3, 1, 0, 43)
    relu2 = my_backend.ReLU()
    pool2 = my_backend.MaxPool(2, 2)
    
    fc = my_backend.FullyConnected(576, NUM_CLASSES, 44)
    loss_fn = my_backend.SoftmaxClassifier()

    # Training Loop
    print(f"\n>>> Measuring data loading time)")
    total_tensor_time = 0

    for flat_pixels, labels, bs in train_loader.get_batch(measure_time=True):
        pass

    for flat_pixels, labels, bs in train_loader.get_batch(measure_time=False):
        # Measure tensor creation separately
        start_tensor = time.perf_counter()
        t = my_backend.Tensor(flat_pixels, [bs, CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
        end_tensor = time.perf_counter()

        total_tensor_time += (end_tensor - start_tensor)


    print(f"Total Tensor Conversion Time: {total_tensor_time:.4f} sec")
    

    print(f"\n>>> Starting Training ({EPOCHS} Epochs)")
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total_samples = 0

        for flat_pixels,labels, bs in train_loader.get_batch():

            # Forward
            t_in = my_backend.Tensor(flat_pixels, [bs, CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
            x = conv1.forward(t_in)
            x = relu1.forward(x)
            x = pool1.forward(x)
            
            x = conv2.forward(x)
            x = relu2.forward(x)
            x = pool2.forward(x)
            
            x = fc.forward(x)
            
            # Loss & Backprop
            probs = loss_fn.predict(x)
            loss, grad = loss_fn.backprop(labels)
            
            # Backward
            grad = fc.backprop(grad, LEARNING_RATE)
            grad = pool2.backprop(grad, LEARNING_RATE)
            grad = relu2.backprop(grad, LEARNING_RATE)
            grad = conv2.backprop(grad, LEARNING_RATE)
            grad = pool1.backprop(grad, LEARNING_RATE)
            grad = relu1.backprop(grad, LEARNING_RATE)
            grad = conv1.backprop(grad, LEARNING_RATE)
            
            # Metrics
            total_loss += loss
            predictions = get_argmax(probs.to_list(), NUM_CLASSES)
            for p, g in zip(predictions, labels):
                if p == g: correct += 1
            total_samples += bs
            
            print(f"Ep {epoch+1} | Loss: {loss:.4f} | Acc: {(correct/total_samples)*100:.1f}%", end='\r')

        epoch_acc = (correct / total_samples) * 100
        print(f"\nEpoch {epoch+1} Done | Avg Loss: {total_loss/total_samples:.4f} | Acc: {epoch_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

    # Save
    save_weights(conv1, conv2, fc)

if __name__ == "__main__":
    main()