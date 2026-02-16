#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include "OutputLayer.hpp"
#include "../utilities/Tensor.hpp"


class SoftmaxClassifier : public OutputLayer {
private:
    Tensor<double> output_;

public:
    SoftmaxClassifier() = default;

    Tensor<double> predict(Tensor<double> input) override;

    std::pair<double, Tensor<double>> backprop(std::vector<int> ground_truth) override;

    Tensor<double> crossEntropyPrime(Tensor<double>& output, std::vector<int>& y);

    double crossEntropy(Tensor<double>& y_hat, std::vector<int>& y);
};