#ifndef CIFAR_EXPERIMENT_RATES_H
#define CIFAR_EXPERIMENT_RATES_H

#include <vector>
#include <map>

#include "network.h"
#include "exp.h"

#include "Eigen/Dense"

class CifarRateExperiment : public Experiment {

public:
    void run() override;

private:

    void readCifarInput(
        std::vector<std::string> &file_names,
        Eigen::MatrixXf& input_buffer,
        Eigen::MatrixXf& label_buffer);


    void runCifar(int trial,
                         int dataset_size,
                         Eigen::MatrixXf& train_input,
                         Eigen::MatrixXf& train_output,
                         Eigen::MatrixXf& test_input,
                         Eigen::MatrixXf& test_output);

    NetworkConfig getConfig();

    std::map<std::string, std::vector<Scenario>> getScenarios(int epoch_count);
};

#endif
