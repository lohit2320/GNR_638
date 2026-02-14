#include "FullyConnected.hpp"
#include <random>
#include <cmath>      // Required for sqrt
#include <stdexcept>  // Required for std::runtime_error
#include "../utilities/Tensor.hpp"

FullyConnected::FullyConnected(int input_size, int output_size, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::vector<int> weight_dims = {input_size, output_size};
    weights_ = Tensor<double>(2,weight_dims);
    weights_.randn(generator,distribution,sqrt(2.0/input_size));
    std::vector<int> bias_dims = {output_size};
    bias_ = Tensor<double>(1,bias_dims);
    bias_.randn(generator,distribution,0);
}

Tensor<double>& FullyConnected::forward(Tensor<double>& input) {
    num_input_dims = input.num_dims;
    input_dims = input.dims;
    if (input.num_dims != 2) {
        int flatten_size = 1;
        for (int i = 1; i < num_input_dims; i++) {
            flatten_size *= input_dims[i];
        }
        std::vector<int> dims = {input.dims[0],flatten_size};
        input.view(2,dims);
    }
    this->input_ = input;
    product_ = input.matmul(weights_)+bias_;
    return product_;
}

Tensor<double> FullyConnected::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> weightGradient = input_.transpose().matmul(chainGradient);
    Tensor<double> biasGradient = chainGradient.col_sum();
    chainGradient = chainGradient.matmul(weights_.transpose());
    chainGradient.view(num_input_dims,input_dims);
    weights_ -= weightGradient*learning_rate;
    bias_ -= biasGradient*learning_rate;
    return chainGradient;
}

void FullyConnected::load(FILE *file_model) {
    double value;
    for (int i = 0; i < weights_.dims[0]; ++i) {
        for (int j = 0; j < weights_.dims[1]; ++j) {
            int read = fscanf(file_model, "%lf", &value);
            if (read != 1) throw std::runtime_error("Invalid model file");
            weights_.set(i, j, value);
        }
    }

    for (int i = 0; i < bias_.dims[0]; ++i) {
        int read = fscanf(file_model, "%lf", &value);
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias_.set(i, value);
    }
}

void FullyConnected::save(FILE *file_model) {
    for (int i = 0; i < weights_.dims[0]; ++i) {
        for (int j = 0; j < weights_.dims[1]; ++j) {
            fprintf(file_model, "%.18lf ", weights_.get(i, j));
        }
    }

    for (int i = 0; i < bias_.dims[0]; ++i) {
        fprintf(file_model, "%.18lf ", bias_.get(i));
    }
}



