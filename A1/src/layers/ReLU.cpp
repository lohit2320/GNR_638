#include "ReLU.hpp"

Tensor<double>& ReLU::forward(Tensor<double>& input) {
    this->input_ = input;
    this->product_ = input.relu();
    return this->product_;
}

Tensor<double> ReLU::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> local_gradient = input_.relu_derivative();
    return chainGradient * local_gradient;
}

void ReLU::load(FILE* file_model) {}

void ReLU::save(FILE* file_model) {}