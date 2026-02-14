#ifndef MAXPOOL_HPP
#define MAXPOOL_HPP

#include "Module.hpp"

class MaxPool : public Module {
private:
    Tensor<double> output_;
    Tensor<double> input_;
    Tensor<int> indexes; // Stores which pixel won (for backprop)
    int stride_, size_;

public:
    
    explicit MaxPool(int size, int stride);

    Tensor<double> &forward(Tensor<double> &input) override;
    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;
    void save(FILE *file_model) override;
};

#endif 