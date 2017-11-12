#ifndef MNIST_DROPOUT_DYNAMIC_EXPERIMENT_H
#define MNIST_DROPOUT_DYNAMIC_EXPERIMENT_H

#include <vector>
#include <map>

#include "exp_mnist.h"
#include "network.h"

#include "Eigen/Dense"

class MnistDynamicExperiment : public MnistExperiment {

public:
    void run() override;

private:

    NetworkConfig getConfig();
};

#endif
