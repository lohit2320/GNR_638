import sys
import os
import time
import numpy as np

# Setup Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import my_backend

def log(msg):
    print(f"[STEP] {msg}...", flush=True) # flush=True forces print immediately

def test_tensor_memory():
    log("1. Allocating small Tensor (Check memory)")
    try:
        # Create 32x32 image, 3 channels, Batch size 1
        t = my_backend.Tensor([0.1]*3072, [1, 3, 32, 32])
        log("   -> Success: Tensor allocated")
    except Exception as e:
        print(f"   -> FAILED: {e}")
        sys.exit(1)

def test_conv2d_forward():
    log("2. Testing Conv2d Forward (Check for infinite loop)")
    # Input: 1 image, 3 channels, 32x32
    # Conv: 3x3 kernel, 1 stride
    t_in = my_backend.Tensor([0.1]*3072, [1, 3, 32, 32])
    layer = my_backend.Conv2d(3, 8, 3, 1, 0)
    
    start = time.time()
    out = layer.forward(t_in)
    end = time.time()
    
    log(f"   -> Success: Forward pass took {end - start:.4f}s")
    # Expected Output shape: 1, 8, 30, 30
    # 30 * 30 * 8 = 7200 elements
    
def test_conv2d_backprop():
    log("3. Testing Conv2d Backprop (The dangerous part)")
    # We need a gradient tensor of the same shape as output: [1, 8, 30, 30]
    # 30x30 = 900. 900*8 = 7200.
    grad_data = [0.1] * 7200
    grad_tensor = my_backend.Tensor(grad_data, [1, 8, 30, 30])
    
    layer = my_backend.Conv2d(3, 8, 3, 1, 0)
    # Must run forward once to cache input
    t_in = my_backend.Tensor([0.1]*3072, [1, 3, 32, 32])
    layer.forward(t_in)
    
    start = time.time()
    # Backprop with learning rate 0.01
    layer.backprop(grad_tensor, 0.01)
    end = time.time()
    
    log(f"   -> Success: Backprop took {end - start:.4f}s")

def test_full_chain():
    log("4. Testing Full Chain (Conv -> ReLU -> Pool -> FC)")
    
    # 1. Input
    x = my_backend.Tensor([0.5]*3072, [1, 3, 32, 32])
    
    # 2. Setup Layers
    conv = my_backend.Conv2d(3, 8, 3, 1, 0)
    relu = my_backend.ReLU()
    pool = my_backend.MaxPool(2, 2)
    fc   = my_backend.FullyConnected(8 * 15 * 15, 10) # 1800 -> 10
    
    # 3. Forward
    log("   -> Running Forward...")
    x = conv.forward(x)
    x = relu.forward(x)
    x = pool.forward(x)
    x = fc.forward(x)
    
    # 4. Backward
    log("   -> Running Backward...")
    # Mock gradient for 10 classes
    grad = my_backend.Tensor([0.1]*10, [1, 10])
    
    grad = fc.backprop(grad, 0.01)
    grad = pool.backprop(grad, 0.01)
    grad = relu.backprop(grad, 0.01)
    grad = conv.backprop(grad, 0.01)
    
    log("   -> Success: Full chain functional")

if __name__ == "__main__":
    print("--- STARTING CRASH DEBUGGER ---")
    test_tensor_memory()
    test_conv2d_forward()
    test_conv2d_backprop()
    test_full_chain()
    print("--- ALL TESTS PASSED ---")
    print("If this script works, the issue is your DataLoader eating all RAM.")