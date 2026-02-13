#pragma once
#include "Module.hpp"
#include "../utilities/Tensor.hpp"

class FullyConnected: public Module {
private:
    Tensor<double> weights_;
    Tensor<double> bias_;
    Tensor<double> input_;
    Tensor<double> product_;
    std::vector<int> input_dims;
    int num_input_dims = -1;
public:
    FullyConnected(int input_size, int output_size, int seed = 0);
    Tensor<double> & forward(Tensor<double> &input) override;
    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;
    void load(FILE *file_model) override;
    void save(FILE *file_model) override;



};

