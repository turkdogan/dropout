#include "exp_mnist_grad.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistGradExperiment::run() {
    std::cout << "Mnist Dropout Grad Experiment Run..." << std::endl;

    int total_size = 60000;

    Eigen::MatrixXf input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(input, output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int validation_start_index = rand() % 59000;

    Eigen::MatrixXf validation_input = input.block(validation_start_index, 0, 1000, input.cols());
    Eigen::MatrixXf validation_output = output.block(validation_start_index, 0, 1000, output.cols());

    int dataset_sizes[] = {200, 1000, 2000, 10000, 20000};
    for (int dataset_size : dataset_sizes) {

        Eigen::MatrixXf train_input = input.block(0, 0, dataset_size, input.cols());
        Eigen::MatrixXf train_output = output.block(0, 0, dataset_size, output.cols());

        NetworkConfig config = getConfig();

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
        training_result.name = scenario_name + "_grad" + std::to_string(dataset_size);
        training_result.category = "Mnist_dropout_overfit";

        std::cout << "write training result... " << std::endl;
        writeTrainingResult(training_result, scenario_name + ".txt", true);
    }
}

NetworkConfig MnistGradExperiment::getConfig() {
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
    config1.scenario = Scenario("Mnist_grad");

    config1.addLayerConfig(dim1, dim2, Activation::Sigmoid, false, true, false);
    config1.addLayerConfig(dim2, dim3, Activation::Softmax, false, false, false);

    return config1;
}
