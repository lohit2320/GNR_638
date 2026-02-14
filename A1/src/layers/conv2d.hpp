#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "Module.hpp"
#include "../utilities/Tensor.hpp"
#include <cstdio>

class Conv2d : public Module {
private:
    Tensor<double> input_;
    Tensor<double> product_;
    int stride, padding;

public:
    Tensor<double> kernels;
    Tensor<double> bias;

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed = 0);

    // Return reference to match Module interface
    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chain_gradient, double learning_rate) override;

    void load(FILE *file_model) override;
    void save(FILE *file_model) override;
};

#endif // CONV2D_HPP