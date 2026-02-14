#pragma once

#include "LRScheduler.hpp"

class LinearLRScheduler : public LRScheduler {
private:
    double step;

public:
    LinearLRScheduler(double initial_lr, double step);

    void onIterationEnd(int iteration) override;
};