import os
import glob
import cv2
import random
import time
import math

class DataLoader:
    def __init__(self, data_dir, batch_size=32, shuffle=True, img_size=(32, 32)):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        self.data = []
        self.classes = []
        self.num_classes = 0
        
        print(f"[DataLoader] Scanning {data_dir}...")
        self._load_dataset()
        print(f"[DataLoader] Loaded {len(self.data)} images.")

    def _load_dataset(self):
        if not os.path.exists(self.data_dir):
            return
            
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                             if os.path.isdir(os.path.join(self.data_dir, d))])
        self.num_classes = len(self.classes)
        
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                files.extend(glob.glob(os.path.join(class_path, ext)))
            
            for file_path in files:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                img = cv2.resize(img, self.img_size)
                flat_img = (img.flatten() / 255.0).tolist()
                self.data.append((flat_img, label_idx))

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
            
        for i in range(0, len(self.data), self.batch_size):
            batch_raw = self.data[i : i + self.batch_size]
            
            batch_inputs = []
            batch_labels = []
            
            for flat_img, label_idx in batch_raw:
                batch_inputs.append(flat_img)
                one_hot = [0.0] * self.num_classes
                one_hot[label_idx] = 1.0
                batch_labels.append(one_hot)
            
            yield batch_inputs, batch_labels