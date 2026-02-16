#ifndef MAXPOOL_HPP
#define MAXPOOL_HPP

#include "Module.hpp"
#include "../utilities/Tensor.hpp"
#include <cstdio> 

class MaxPool : public Module {
private:
    int stride_;
    int size_;
    
    Tensor<double> input_cache_;
    Tensor<double> output_; 
    Tensor<double> indexes_; 

public:
    MaxPool(int size, int stride);

    Tensor<double>& forward(Tensor<double>& input) override;
    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override {}
    void save(FILE *file_model) override {}
};

#endif