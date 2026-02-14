import os
import time
import cv2  
import random

class DataLoader:
    def __init__(self, data_dir, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
      
        self.classes = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
       
        start_time = time.time()
        self.image_paths = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        
        self.disk_scan_time = time.time() - start_time
        self.num_classes = len(self.classes)

    def get_batch_generator(self):
        if self.shuffle:
            random.shuffle(self.image_paths)
            
        for i in range(0, len(self.image_paths), self.batch_size):
            batch_data = self.image_paths[i : i + self.batch_size]
            batch_x = []
            batch_y = []
            
            for path, label in batch_data:
               
                img = cv2.imread(path)
                img = cv2.resize(img, (32, 32))
                
               
                pixels = []
                for c in range(3):
                    for h in range(32):
                        for w in range(32):
                            pixels.append(img[h, w, c] / 255.0)
                
                batch_x.extend(pixels)
                batch_y.append(label)
                
            yield batch_x, batch_y

    def __len__(self):
        return len(self.image_paths) // self.batch_size