#ifndef MNIST_OVERFIT_EXPERIMENT_H
#define MNIST_OVERFIT_EXPERIMENT_H

#include <vector>

#include "exp_mnist.h"
#include "network.h"

#include "Eigen/Dense"

class MnistOverfitExperiment : public MnistExperiment {

public:
    void run() override;

private:

    std::vector<NetworkConfig> getConfigs();
};

#endif
