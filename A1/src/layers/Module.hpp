#ifndef MODULE_HPP
#define MODULE_HPP

#include "../utilities/Tensor.hpp"


class Module {
protected:
    bool isEval = false;

public:

    virtual Tensor<float> forward(Tensor<float> &input) = 0;
    virtual Tensor<float> backprop(Tensor<float> &chainGradient, float learning_rate) = 0;
    void train() { this->isEval = false; }
    void eval()  { this->isEval = true; }

    virtual ~Module() = default;
};

#endif