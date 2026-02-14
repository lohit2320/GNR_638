#include "LinearLRScheduler.hpp"

LinearLRScheduler::LinearLRScheduler(double initial_lr, double step) {
    this->learning_rate = initial_lr; 
    this->step = step;
}

void LinearLRScheduler::onIterationEnd(int iteration) {
    this->learning_rate += this->step;
}