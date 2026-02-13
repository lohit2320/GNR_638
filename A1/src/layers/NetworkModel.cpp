#include "NetworkModel.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm> // for std::max

using namespace std;

NetworkModel::NetworkModel(std::vector<Module*>& modules, OutputLayer* output_layer, LRScheduler* lr_scheduler) {
    this->modules_ = modules;
    this->output_layer_ = output_layer;
    this->lr_scheduler_ = lr_scheduler;
    this->iteration = 0;
}

double NetworkModel::trainStep(Tensor<double>& x, std::vector<int>& y) {
    // 1. Forward Pass
    Tensor<double> output = this->forward(x);

    // 2. Backpropagation on Output Layer
    pair<double, Tensor<double>> loss_and_grad = output_layer_->backprop(y);
    Tensor<double> chain_gradient = loss_and_grad.second;

    // 3. Backpropagation through hidden layers (in reverse)
    double current_lr = lr_scheduler_->learning_rate;
    for (int i = (int)modules_.size() - 1; i >= 0; --i) {
        chain_gradient = modules_[i]->backprop(chain_gradient, current_lr);
    }

    // 4. Update Scheduler
    ++iteration;
    lr_scheduler_->onIterationEnd(iteration);

    return loss_and_grad.first;
}

Tensor<double> NetworkModel::forward(Tensor<double>& x) {
    Tensor<double> activation = x; 
    
    for (auto& module : modules_) {
        activation = module->forward(activation);
    }
    
    return output_layer_->predict(activation);
}

std::vector<int> NetworkModel::predict(Tensor<double>& x) {
    Tensor<double> output = this->forward(x);
    
    std::vector<int> predictions;
    predictions.reserve(output.dims[0]);

    for (int i = 0; i < output.dims[0]; ++i) {
        int argmax = -1;
        double max_val = -1e9; 
        
        for (int j = 0; j < output.dims[1]; ++j) {
            double val = output.get(i, j);
            if (val > max_val) {
                max_val = val;
                argmax = j;
            }
        }
        predictions.push_back(argmax);
    }

    return predictions;
}

void NetworkModel::load(std::string path) {
    FILE* model_file = fopen(path.c_str(), "rb");
    if (!model_file) {
        throw std::runtime_error("Error reading model file: " + path);
    }
    
    for (auto& module : modules_) {
        module->load(model_file);
    }
    
    fclose(model_file);
}

void NetworkModel::save(std::string path) {
    FILE* model_file = fopen(path.c_str(), "wb");
    if (!model_file) {
        throw std::runtime_error("Error writing model file: " + path);
    }
    
    for (auto& module : modules_) {
        module->save(model_file);
    }
    
    fclose(model_file);
}

void NetworkModel::eval() {
    for (auto& module : modules_) {
        module->eval();
    }
}

void NetworkModel::train() {
    for (auto& module : modules_) {
        module->train();
    }
}

NetworkModel::~NetworkModel() {
   
}