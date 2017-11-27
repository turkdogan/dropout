#ifndef MNIST_GRAD_EXPERIMENT_H
#define MNIST_GRAD_EXPERIMENT_H

#include <vector>

#include "exp_mnist.h"
#include "network.h"

#include "Eigen/Dense"

class MnistGradExperiment : public MnistExperiment {

public:
    void run() override;

private:

    NetworkConfig getConfig();
};

#endif
