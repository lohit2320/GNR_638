#include "Sigmoid.hpp"

Tensor<double>& Sigmoid::forward(Tensor<double>& input) {
    this->input_ = input;
    this->product_ = input.sigmoid();
    return this->product_;
}

Tensor<double> Sigmoid::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> local_gradient = input_.sigmoid_derivative();
    return chainGradient * local_gradient;
}

void Sigmoid::load(FILE* file_model) {}

void Sigmoid::save(FILE* file_model) {}