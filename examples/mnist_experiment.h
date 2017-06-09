#ifndef MNIST_DROPOUT_EXPERIMENT_H
#define MNIST_DROPOUT_EXPERIMENT_H

#include <vector>

#include "../experiment.h"
#include "../network.h"

#include "../Eigen/Dense"

class MnistExperiment : public Experiment {

public:
    void run() override;

private:
    int reverseInt(int i);

    Eigen::MatrixXf readMnistInput(const std::string& path,
                                   int number_of_items = 60000);

    Eigen::MatrixXf readMnistOutput(const std::string& path,
                                    int number_of_items = 60000);

    void runMnistNetwork(int trial,
                         int dataset_size,
                         Eigen::MatrixXf& train_input,
                         Eigen::MatrixXf& train_output,
                         Eigen::MatrixXf& validation_input,
                         Eigen::MatrixXf& validation_output,
                         Eigen::MatrixXf& test_input,
                         Eigen::MatrixXf& test_output,
                         std::ofstream& out_file);

    NetworkConfig getConfig();

	std::vector<DropoutScenario> getScenarios(int epoch_count);
};

#endif
