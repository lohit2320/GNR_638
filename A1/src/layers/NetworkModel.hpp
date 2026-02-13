#pragma once

#include <vector>
#include <string>


#include "../utilities/Tensor.hpp"
#include "Module.hpp"
#include "OutputLayer.hpp"
#include "LRScheduler.hpp" 


class NetworkModel {
private:
    std::vector<Module*> modules_;
    OutputLayer* output_layer_;
    LRScheduler* lr_scheduler_;
    int iteration = 0;

public:
    NetworkModel(std::vector<Module*>& modules, OutputLayer* output_layer, LRScheduler* lr_scheduler);

    double trainStep(Tensor<double>& x, std::vector<int>& y);

    Tensor<double> forward(Tensor<double>& x);

    std::vector<int> predict(Tensor<double>& x);

    void load(std::string path);

    void save(std::string path);

    void eval();
    void train();

    virtual ~NetworkModel();
};