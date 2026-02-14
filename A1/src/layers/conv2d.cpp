#include "conv2d.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Initialize Kernels (Filters)
    // Shape: [Out_Channels, In_Channels, Kernel_H, Kernel_W]
    std::vector<int> kernel_dims = {out_channels, in_channels, kernel_size, kernel_size};
    kernels = Tensor<double>(4, kernel_dims);
    // He Initialization
    kernels.randn(generator, distribution, sqrt(2.0 / (kernel_size * kernel_size * out_channels)));

    // Initialize Bias
    // Shape: [Out_Channels]
    std::vector<int> bias_dims = {out_channels};
    bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);

    this->stride = stride;
    this->padding = padding;
}

Tensor<double> &Conv2d::forward(Tensor<double> &input) {
    this->input_ = input;
    // NOTE: This assumes your Tensor class has a 'convolve2d' method implemented.
    // If not, you will need to implement the forward pass loop manually like the backprop below.
    product_ = input.convolve2D(kernels, stride, padding, bias);
    return product_;
}

Tensor<double> Conv2d::backprop(Tensor<double> chain_gradient, double learning_rate) {
    Tensor<double> kernels_gradient(kernels.num_dims, kernels.dims);
    Tensor<double> input_gradient(input_.num_dims, input_.dims);
    Tensor<double> bias_gradient(1, bias.dims);
    
    kernels_gradient.zero();
    input_gradient.zero();
    bias_gradient.zero();

    // Manual convolution backprop
    for (int i = 0; i < input_.dims[0]; ++i) { // Batch
        for (int f = 0; f < kernels.dims[0]; f++) { // Filter (Out Channel)
            int x = -padding;
            for (int cx = 0; cx < chain_gradient.dims[2]; x += stride, cx++) { // Output X
                int y = -padding;
                for (int cy = 0; cy < chain_gradient.dims[3]; y += stride, cy++) { // Output Y
                    
                    // Gradient from the next layer for this pixel
                    // NOTE: Ensure your Tensor class supports 4D get()
                    double chain_grad = chain_gradient.get(i, f, cx, cy);
                    
                    bias_gradient.add(f, chain_grad); // Accumulate bias gradient

                    for (int fx = 0; fx < kernels.dims[2]; fx++) { // Kernel X
                        int ix = x + fx; 
                        if (ix >= 0 && ix < input_.dims[2]) {
                            for (int fy = 0; fy < kernels.dims[3]; fy++) { // Kernel Y
                                int iy = y + fy; 
                                if (iy >= 0 && iy < input_.dims[3]) {
                                    for (int fc = 0; fc < kernels.dims[1]; fc++) { // In Channel
                                        
                                        // Update Kernel Gradient
                                        double val_input = input_.get(i, fc, ix, iy);
                                        kernels_gradient.add(f, fc, fx, fy, val_input * chain_grad);
                                        
                                        // Update Input Gradient (to pass back)
                                        double val_kernel = kernels.get(f, fc, fx, fy);
                                        input_gradient.add(i, fc, ix, iy, val_kernel * chain_grad);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    kernels -= kernels_gradient * learning_rate;
    bias -= bias_gradient * learning_rate;

    return input_gradient;
}

void Conv2d::load(FILE *file_model) {
    double value;
    // Load Kernels
    for (int i = 0; i < kernels.dims[0]; ++i) {
        for (int j = 0; j < kernels.dims[1]; ++j) {
            for (int k = 0; k < kernels.dims[2]; ++k) {
                for (int l = 0; l < kernels.dims[3]; ++l) {
                    if (fscanf(file_model, "%lf", &value) != 1) 
                        throw std::runtime_error("Invalid model file");
                    kernels.set(i, j, k, l, value);
                }
            }
        }
    }
    // Load Bias
    for (int m = 0; m < bias.dims[0]; ++m) {
        if (fscanf(file_model, "%lf", &value) != 1) 
            throw std::runtime_error("Invalid model file");
        bias.set(m, value);
    }
}

void Conv2d::save(FILE *file_model) {
    // Save Kernels
    for (int i = 0; i < kernels.dims[0]; ++i) {
        for (int j = 0; j < kernels.dims[1]; ++j) {
            for (int k = 0; k < kernels.dims[2]; ++k) {
                for (int l = 0; l < kernels.dims[3]; ++l) {
                    fprintf(file_model, "%.18lf ", kernels.get(i, j, k, l));
                }
            }
        }
    }
    // Save Bias
    for (int m = 0; m < bias.dims[0]; ++m) {
        fprintf(file_model, "%.18lf ", bias.get(m));
    }
}