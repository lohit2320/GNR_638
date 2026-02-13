#ifndef MAXPOOL_HPP
#define MAXPOOL_HPP

#include "Module.hpp"

class MaxPool : public Module {
private:
    Tensor<float> output_;
    Tensor<float> input_;
    Tensor<int> indexes; // Stores which pixel won (for backprop)
    int stride_, size_;

public:
    
    explicit MaxPool(int size, int stride);

    
    Tensor<float> forward(Tensor<float> &input) override;
    Tensor<float> backprop(Tensor<float> &chainGradient, float learning_rate) override;
};

#endif 