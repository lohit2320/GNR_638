#pragma once

class LRScheduler {
public:
    double learning_rate;

    virtual void onIterationEnd(int iteration) = 0;

    virtual ~LRScheduler() = default;
};