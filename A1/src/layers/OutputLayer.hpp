#pragma once

#include <vector>
#include <utility>
#include "Tensor.hpp"


class OutputLayer {
public:
    virtual Tensor<double> predict(Tensor<double> input) = 0;

    virtual std::pair<double, Tensor<double>> backprop(std::vector<int> ground_truth) = 0;

    virtual ~OutputLayer() = default;
};