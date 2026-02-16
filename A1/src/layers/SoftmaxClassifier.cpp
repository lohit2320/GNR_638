#include "SoftmaxClassifier.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

Tensor<double> SoftmaxClassifier::predict(Tensor<double> input) {
    // 1. Calculate Softmax
    this->output_ = input.softmax();
    return this->output_;
}

std::pair<double, Tensor<double>> SoftmaxClassifier::backprop(std::vector<int> ground_truth) {
    // 2. Calculate Loss and Gradient using the stored output from predict()
    double loss = crossEntropy(this->output_, ground_truth);
    Tensor<double> gradient = crossEntropyPrime(this->output_, ground_truth);

    return std::make_pair(loss, gradient);
}

Tensor<double> SoftmaxClassifier::crossEntropyPrime(Tensor<double>& output, std::vector<int>& y) {
    // Gradient of Softmax + Cross Entropy is simply: (predicted - actual)
    Tensor<double> prime = output; 
    
    for (int i = 0; i < y.size(); ++i) {
        double current_val = prime.get(i, y[i]);
        // Subtract 1.0 from the correct class index (effectively y_hat - 1)
        prime.set(i, y[i], current_val - 1.0);
    } // <--- THIS WAS MISSING
    
    // Average the gradient over the batch size
    return prime / static_cast<double>(output.dims[0]);
}

double SoftmaxClassifier::crossEntropy(Tensor<double>& y_hat, std::vector<int>& y) {
    double total = 0;
    for (int i = 0; i < y.size(); ++i) {
        double x = y_hat.get(i, y[i]);
        // Clip value to avoid log(0) = -inf
        double val = (x < 1e-10) ? 1e-10 : x;
        total += -std::log(val);
    }

    return total / static_cast<double>(y.size());
}