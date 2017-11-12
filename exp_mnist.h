#ifndef MNIST_DROPOUT_EXPERIMENT_H
#define MNIST_DROPOUT_EXPERIMENT_H

#include <vector>
#include <map>

#include "exp.h"
#include "network.h"

#include "Eigen/Dense"

class MnistExperiment : public Experiment {

public:

protected:

    Eigen::MatrixXf readMnistInput(const std::string& path,
                                   int number_of_items = 60000);

    Eigen::MatrixXf readMnistOutput(const std::string& path,
                                    int number_of_items = 60000);


private:
    int reverseInt(int i);

};

#endif
