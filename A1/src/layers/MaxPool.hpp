#ifndef MAXPOOL_HPP
#define MAXPOOL_HPP

#include "Module.hpp"
#include "../utilities/Tensor.hpp"
#include <cstdio> 

class MaxPool : public Module {
private:
    int stride_;
    int size_;
    
    // We MUST store these as members so we can return references to them
    Tensor<double> input_cache_;
    Tensor<double> output_; 
    Tensor<double> indexes_; // Stores max indices

public:
    MaxPool(int size, int stride);

    // Return reference to match Module.hpp
    Tensor<double>& forward(Tensor<double>& input) override;
    
    // Backprop usually returns by value (new tensor), which is fine
    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override {}
    void save(FILE *file_model) override {}
};

#endif