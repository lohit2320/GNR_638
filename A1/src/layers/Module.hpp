#ifndef MODULE_HPP
#define MODULE_HPP

#include "../utilities/Tensor.hpp"


class Module {
protected:
    bool isEval = false;

public:

    virtual Tensor<double> &forward(Tensor<double> &input) = 0;
    virtual Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) = 0;
    void train() { this->isEval = false; }
    void eval()  { this->isEval = true; }

    virtual void load(FILE *file_model) = 0;
    virtual void save(FILE *file_model) = 0;

    virtual ~Module() = default;
};

#endif