#include "SoftmaxClassifier.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

Tensor<double> SoftmaxClassifier::predict(Tensor<double> input) {
    this->output_ = input.softmax();
    return this->output_;
}

std::pair<double, Tensor<double>> SoftmaxClassifier::backprop(std::vector<int> ground_truth) {
    double loss = crossEntropy(this->output_, ground_truth);
    Tensor<double> gradient = crossEntropyPrime(this->output_, ground_truth);

    return std::make_pair(loss, gradient);
}

Tensor<double> SoftmaxClassifier::crossEntropyPrime(Tensor<double>& output, std::vector<int>& y) {
    Tensor<double> prime = output; 
    
    // Gradient of Softmax + CrossEntropy is (p - y)
    // We subtract 1 from the probability of the correct class
    for (int i = 0; i < y.size(); ++i) {
        double current_val = prime.get(i, y[i]);
        prime.set(i, y[i], current_val - 1.0);
    }

    // Normalize by batch size
    return prime / static_cast<double>(output.dims[0]);
}

double SoftmaxClassifier::crossEntropy(Tensor<double>& y_hat, std::vector<int>& y) {
    double total = 0;
    for (int i = 0; i < y.size(); ++i) {
        double x = y_hat.get(i, y[i]);
        double val = (x < 1e-10) ? 1e-10 : x;
        total += -std::log(val);
    }

    return total / static_cast<double>(y.size());
}