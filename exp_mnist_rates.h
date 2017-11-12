#ifndef MNIST_DROPOUT_RATES_EXPERIMENT_H
#define MNIST_DROPOUT_RATES_EXPERIMENT_H

#include <vector>
#include <map>

#include "exp_mnist.h"
#include "network.h"

#include "Eigen/Dense"

class MnistRateExperiment : public MnistExperiment {

public:
    void run() override;

private:

    NetworkConfig getConfig();
};

#endif
