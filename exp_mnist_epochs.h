#ifndef MNIST_DROPOUT_EPOCHS_EXPERIMENT_H
#define MNIST_DROPOUT_EPOCHS_EXPERIMENT_H

#include <vector>
#include <map>

#include "exp_mnist.h"
#include "network.h"

#include "Eigen/Dense"

class MnistEpochExperiment : public MnistExperiment {

public:
    void run() override;

private:

    NetworkConfig getConfig();
};

#endif
