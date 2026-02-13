#include "MaxPool.hpp"
#include <limits>
#include <vector>

MaxPool::MaxPool(int size, int stride) : size_(size), stride_(stride) {}

Tensor<float> MaxPool::forward(Tensor<float> &input) {
    this->input_ = input; // Save input for shape reference
    
    // Assume input shape is [N, C, H, W]
    int N = input.dims[0];
    int C = input.dims[1];
    int H = input.dims[2];
    int W = input.dims[3];

    int H_out = (H - size_) / stride_ + 1;
    int W_out = (W - size_) / stride_ + 1;

    std::vector<int> out_dims = {N, C, H_out, W_out};
    
    // Resize output and index tensors
    output_ = Tensor<float>(4, out_dims);
    indexes = Tensor<int>(4, out_dims);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    
                    int h_start = h * stride_;
                    int w_start = w * stride_;
                    
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1; // Will store flattened index (h_in * W_in + w_in)

                    for (int i = 0; i < size_; ++i) {
                        for (int j = 0; j < size_; ++j) {
                            int cur_h = h_start + i;
                            int cur_w = w_start + j;

                            // Boundary check
                            if (cur_h < H && cur_w < W) {
                                float val = input.get(n, c, cur_h, cur_w);
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = cur_h * W + cur_w; 
                                }
                            }
                        }
                    }
                    output_.set(n, c, h, w, max_val);
                    indexes.set(n, c, h, w, max_idx);
                }
            }
        }
    }
    return output_;
}

Tensor<float> MaxPool::backprop(Tensor<float> &chainGradient, float learning_rate) {
    // Gradient input has same shape as original input
    Tensor<float> grad_input(input_.num_dims, input_.dims);
    grad_input.zero();

    int N = output_.dims[0];
    int C = output_.dims[1];
    int H_out = output_.dims[2];
    int W_out = output_.dims[3];
    int W_in = input_.dims[3];

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    
                    // Retrieve the index of the max value from forward pass
                    int max_idx = indexes.get(n, c, h, w);
                    float grad = chainGradient.get(n, c, h, w);
                    
                    if (max_idx != -1) {
                        // Convert flat index back to 2D coordinates
                        int h_in = max_idx / W_in;
                        int w_in = max_idx % W_in;
                        
                        // Accumulate gradient (handles cases where windows might overlap)
                        float current_val = grad_input.get(n, c, h_in, w_in);
                        grad_input.set(n, c, h_in, w_in, current_val + grad);
                    }
                }
            }
        }
    }
    
    // MaxPool has no learnable parameters, so we just return the gradient w.r.t input
    return grad_input;
}