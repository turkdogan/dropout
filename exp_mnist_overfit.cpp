#include "exp_mnist_overfit.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistOverfitExperiment::run() {
    std::cout << "Mnist Dropout Overfit Experiment Run..." << std::endl;

    int total_size = 60000;

    Eigen::MatrixXf input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(input, output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int validation_start_index = rand() % 59000;

    Eigen::MatrixXf validation_input = input.block(validation_start_index, 0, 1000, input.cols());
    Eigen::MatrixXf validation_output = output.block(validation_start_index, 0, 1000, output.cols());

    int dataset_sizes[] = {1000, 2000, 10000, 60000};
    for (int dataset_size : dataset_sizes) {

        Eigen::MatrixXf train_input = input.block(0, 0, dataset_size, input.cols());
        Eigen::MatrixXf train_output = output.block(0, 0, dataset_size, output.cols());

        std::vector<NetworkConfig> configs = getConfigs();

        for (NetworkConfig config : configs) {
            srand(99);

            Network network(config);

            TrainingResult training_result = network.trainNetwork(
                train_input, train_output,
                validation_input, validation_output,
                false);

            int correct = network.test(test_input, test_output);
            training_result.count = 10000;
            training_result.correct = correct;
            training_result.trial = 1;
            training_result.dataset_size = total_size;
            training_result.correct = correct;
            std::string scenario_name =
                std::to_string(dataset_size) + "_" +
                config.scenario.name();
            training_result.name = scenario_name + "_overfit" + std::to_string(dataset_size);
            training_result.category = "Mnist_dropout_overfit";

            std::cout << "write training result... " << std::endl;
            writeTrainingResult(training_result, scenario_name + ".txt", true);
        }
    }
}

std::vector<NetworkConfig> MnistOverfitExperiment::getConfigs() {
    const int dim1 = 784;
    const int dim2 = 200;
    const int dim3 = 10;

    NetworkConfig config1;
    config1.epoch_count = 120;
    config1.report_each = 2;
    config1.batch_size = 40;
    config1.momentum = 0.9f;
    config1.learning_rate = 0.01f;
    config1.clip_before_error = true;
    config1.scenario = Scenario("Mnist_no_drop_overfit");

    config1.addLayerConfig(dim1, dim2, Activation::Sigmoid, false, false, false);
    config1.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);

    NetworkConfig config2;
    config2.epoch_count = 120;
    config2.report_each = 2;
    config2.batch_size = 40;
    config2.momentum = 0.9f;
    config2.learning_rate = 0.01f;
    config2.clip_before_error = true;
    config2.scenario = Scenario("Mnist_drop05_overfit", config2.epoch_count, 0.5f);

    config2.addLayerConfig(dim1, dim2, Activation::Sigmoid, true, false, false);
    config2.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);

    std::vector<NetworkConfig> configs;
    configs.push_back(config1);
    configs.push_back(config2);

    return configs;
}
