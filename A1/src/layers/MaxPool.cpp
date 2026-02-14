#include "MaxPool.hpp"
#include <limits>
#include <vector>
#include <cmath> 

MaxPool::MaxPool(int size, int stride) : stride_(stride), size_(size) {}

Tensor<double>& MaxPool::forward(Tensor<double> &input) {
    this->input_cache_ = input; 

    int N = input.dims[0];
    int C = input.dims[1];
    int H = input.dims[2];
    int W = input.dims[3];

    int H_out = (H - size_) / stride_ + 1;
    int W_out = (W - size_) / stride_ + 1;

    std::vector<int> out_dims = {N, C, H_out, W_out};

 
    this->output_ = Tensor<double>(4, out_dims);
    this->indexes_ = Tensor<double>(4, out_dims);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    
                    int h_start = h * stride_;
                    int w_start = w * stride_;
                    
                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_idx = -1; 

                    for (int i = 0; i < size_; ++i) {
                        for (int j = 0; j < size_; ++j) {
                            int cur_h = h_start + i;
                            int cur_w = w_start + j;

                            if (cur_h < H && cur_w < W) {
                                int flat_idx = n*(C*H*W) + c*(H*W) + cur_h*W + cur_w;
                                double val = input.get(flat_idx);
                                
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = cur_h * W + cur_w; 
                                }
                            }
                        }
                    }
                    
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + h*W_out + w;
                    
                    this->output_.set(out_idx, max_val);
                    this->indexes_.set(out_idx, (double)max_idx);
                }
            }
        }
    }
    // Return reference to the member variable
    return this->output_;
}

Tensor<double> MaxPool::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> grad_input(input_cache_.dims.size(), input_cache_.dims);
    grad_input.zero();

    if (chainGradient.dims.size() < 4) return grad_input;

    int N = chainGradient.dims[0];
    int C = chainGradient.dims[1];
    int H_out = chainGradient.dims[2];
    int W_out = chainGradient.dims[3];
    int W_in = input_cache_.dims[3];
    int H_in = input_cache_.dims[2];

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + h*W_out + w;
                    
                    int max_idx = (int)indexes_.get(out_idx);
                    double grad = chainGradient.get(out_idx);
                    
                    if (max_idx != -1) {
                        int h_in = max_idx / W_in;
                        int w_in = max_idx % W_in;
                        
                        int in_flat_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + h_in*W_in + w_in;

                        double current_val = grad_input.get(in_flat_idx);
                        grad_input.set(in_flat_idx, current_val + grad);
                    }
                }
            }
        }
    }
    return grad_input;
}