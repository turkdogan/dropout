#include "exp_mnist_activations.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistActivationExperiment::run() {
    std::cout << "Mnist Dropout Activations Experiment Run..." << std::endl;

    int total_size = 60000;

    // 60k sample input
    // 10k sample ouput
    Eigen::MatrixXf train_input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf train_output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(train_input, train_output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int epochs[] = {10, 20, 30};

    Activation activations[] = {
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::ReLU
    };

    for (Activation &activation : activations) {

        NetworkConfig config = getConfig(activation);
        Scenario scenario("C0.5_" + std::to_string(config.epoch_count), config.epoch_count, 0.5f);
        config.scenario = scenario;

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
            scenario.name();
        training_result.name = scenario_name;
        // TODO update category here...
        training_result.category = "Mnist_activations";

        std::cout << "write training result... " << std::endl;
        writeTrainingResult(training_result, scenario_name + ".txt", false);
    }

}

NetworkConfig MnistActivationExperiment::getConfig(Activation activation) {

    const int dim1 = 784;
    const int dim2 = 400;
    const int dim3 = 100;
    const int dim4 = 10;

    NetworkConfig config;
    // will be updated before training
    config.epoch_count = 120;
    config.report_each = 2;
    config.batch_size = 40;
    config.momentum = 0.9f;
    config.learning_rate = 0.01f;
    config.clip_before_error = (activation == Activation::ReLU);

    config.addLayerConfig(dim1, dim2, activation, true, false, false);
    config.addLayerConfig(dim2, dim3, activation, true, false, false);
    config.addLayerConfig(dim3, dim4, Activation::Softmax, false, false, false);

    return config;
}
