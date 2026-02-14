#pragma once

#include <cstdio>
#include "../utilities/Tensor.hpp"
#include "Module.hpp"

class ReLU : public Module {
private:
    Tensor<double> input_;
    Tensor<double> product_;

public:
    ReLU() = default;

    Tensor<double>& forward(Tensor<double>& input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE* file_model) override;

    void save(FILE* file_model) override;
};