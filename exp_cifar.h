#ifndef CIFAR_EXPERIMENT_H
#define CIFAR_EXPERIMENT_H

#include <vector>
#include <map>

#include "exp.h"
#include "network.h"

#include "Eigen/Dense"

class CifarExperiment : public Experiment {

public:
    void run() override;

private:

    void readCifarInput(
        const std::string& path,
        Eigen::MatrixXf& input_buffer,
        Eigen::MatrixXf& label_buffer,
        int number_of_items = 1000,
        int starting_index = 0);


    void runCifar(int trial,
                         int dataset_size,
                         Eigen::MatrixXf& train_input,
                         Eigen::MatrixXf& train_output,
                         Eigen::MatrixXf& validation_input,
                         Eigen::MatrixXf& validation_output,
                         Eigen::MatrixXf& test_input,
                         Eigen::MatrixXf& test_output);

    NetworkConfig getConfig();

    std::map<std::string, std::vector<Scenario>> getScenarios(int epoch_count);
};

#endif
