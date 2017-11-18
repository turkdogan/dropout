#include "exp_mnist_params.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistParamExperiment::run() {
    std::cout << "Mnist Dropout Params Experiment Run..." << std::endl;

    int total_size = 60000;

    // 60k sample input
    // 10k sample ouput
    Eigen::MatrixXf train_input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf train_output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(train_input, train_output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int params[] = {10, 100, 300, 800};

    srand(99);

    for (int param : params) {
        NetworkConfig config = getConfig(param);
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
            std::to_string(param) + "_" +
            config.scenario.name();
        training_result.name = scenario_name;
        // TODO update category here...
        training_result.category = "Mnist_params";

        std::cout << "write training result... " << std::endl;
        writeTrainingResult(training_result, scenario_name + ".txt", false);
    }

}

NetworkConfig MnistParamExperiment::getConfig(int params) {
    const int dim1 = 784;
    const int dim2 = params;
    const int dim3 = 10;

    NetworkConfig config;
    // will be updated before training
    config.epoch_count = 120;
    config.report_each = 5;
    config.batch_size = 40;
    config.momentum = 0.9f;
    config.learning_rate = 0.01f;

    config.addLayerConfig(dim1, dim2, Activation::Sigmoid, true, false, false);
    config.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);

    config.scenario = Scenario("C0.5_params", config.epoch_count, 0.5f);
    return config;
}
