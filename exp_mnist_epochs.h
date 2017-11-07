#ifndef MNIST_DROPOUT_EPOCHS_EXPERIMENT_H
#define MNIST_DROPOUT_EPOCHS_EXPERIMENT_H

#include <vector>
#include <map>

#include "exp.h"
#include "network.h"

#include "Eigen/Dense"

class MnistEpochExperiment : public Experiment {

public:
    void run() override;

private:
    int reverseInt(int i);

    Eigen::MatrixXf readMnistInput(const std::string& path,
                                   int number_of_items = 60000);

    Eigen::MatrixXf readMnistOutput(const std::string& path,
                                    int number_of_items = 60000);

    NetworkConfig getConfig();
};

#endif
