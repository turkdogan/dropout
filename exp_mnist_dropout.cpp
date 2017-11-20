#include "exp_mnist_dropout.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistDropoutExperiment::run() {
    std::cout << "Mnist Dropout Experiment Run..." << std::endl;

    int total_size = 60000;

    Eigen::MatrixXf train_input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf train_output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(train_input, train_output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    std::vector<NetworkConfig> configs = getConfigs();

    for (NetworkConfig config : configs) {
        srand(99);

        Network network(config);

        TrainingResult training_result = network.trainNetwork(train_input, train_output);

        std::cout << "training result..." << std::endl;
        int correct = network.test(test_input, test_output);
        training_result.count = 10000;
        training_result.correct = correct;
        training_result.trial = 1;
        training_result.dataset_size = total_size;
        training_result.correct = correct;
        std::string scenario_name =
            std::to_string(total_size) + "_" +
            config.scenario.name();
        training_result.name = scenario_name;
        // TODO update category here...
        training_result.category = "Mnist_dropout";

        std::cout << "write training result... " << std::endl;
        writeTrainingResult(training_result, scenario_name + ".txt", false);
    }

}

std::vector<NetworkConfig> MnistDropoutExperiment::getConfigs() {


    const int dim1 = 784;
    const int dim2 = 200;
    const int dim3 = 10;

    NetworkConfig config1;
    config1.epoch_count = 120;
    config1.report_each = 2;
    config1.batch_size = 40;
    config1.momentum = 0.9f;
    config1.learning_rate = 0.01f;
    config1.clip_before_error = false;
    config1.scenario = Scenario("Mnist_no_drop");

    config1.addLayerConfig(dim1, dim2, Activation::Sigmoid, false, false, false);
    config1.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);

    NetworkConfig config2;
    config2.epoch_count = 120;
    config2.report_each = 2;
    config2.batch_size = 40;
    config2.momentum = 0.9f;
    config2.learning_rate = 0.01f;
    config2.clip_before_error = false;
    config2.scenario = Scenario("Mnist_drop05", config2.epoch_count, 0.5f);

    config2.addLayerConfig(dim1, dim2, Activation::Sigmoid, true, false, false);
    config2.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);

    NetworkConfig config3;
    config3.epoch_count = 120;
    config3.report_each = 2;
    config3.batch_size = 40;
    config3.momentum = 0.9f;
    config3.learning_rate = 0.01f;
    config3.clip_before_error = false;

    config3.addLayerConfig(dim1, dim2, Activation::Sigmoid, true, false, false);
    config3.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);
    config3.scenario = Scenario("Mnist_drop07", config3.epoch_count, 0.7f);


    std::vector<NetworkConfig> configs;
    configs.push_back(config1);
    configs.push_back(config2);
    configs.push_back(config3);

    return configs;
}
