#include "exp_mnist_epochs.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistEpochExperiment::run() {
    std::cout << "Mnist Dropout Epochs Experiment Run..." << std::endl;

    int total_size = 60000;

    // 60k sample input
    // 10k sample ouput
    Eigen::MatrixXf train_input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf train_output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(train_input, train_output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int epochs[] = {10, 20, 30};

    NetworkConfig config = getConfig();

    for (int epoch : epochs) {
        Scenario scenario("C0.5_" + std::to_string(epoch), epoch, 0.5f);
        config.scenario = scenario;

        config.epoch_count = epoch;

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
        training_result.category = "Mnist_epochs";

        std::cout << "write training result... " << std::endl;
        writeTrainingResult(training_result, scenario_name + ".txt", false);
    }

}

int MnistEpochExperiment::reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Eigen::MatrixXf MnistEpochExperiment::readMnistInput(const std::string& path,
                                                       int number_of_items)
{
    std::ifstream file(path, std::ios::binary);
    Eigen::MatrixXf result(number_of_items, 784);

    if (!file.is_open()) {
        std::cerr << "MNIST data file could not be read" << std::endl;
        return result;
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    for (int i = 0; i < number_of_items; i++) {
        for (int j = 0; j < n_rows * n_cols; j++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            result(i, j) = (float)temp / 255.0f;
        }
    }
    return result;
}

Eigen::MatrixXf MnistEpochExperiment::readMnistOutput(const std::string& path,
                                                        int number_of_items)
{
    float one_hot_map[10][10] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    Eigen::MatrixXf result(number_of_items, 10);

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "MNIST data file could not be read" << std::endl;
        return result;
    }

    int magic_number = 0;
    int number_of_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverseInt(number_of_labels);

    for (int i = 0; i < number_of_items; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        int index = (int)temp;
        for (int j = 0; j < 10; j++) {
            result(i, j) = one_hot_map[index][j];
        }
    }
    return result;
}

NetworkConfig MnistEpochExperiment::getConfig() {
    const int dim1 = 784;
    const int dim2 = 200;
    const int dim3 = 100;
    const int dim4 = 50;
    const int dim5 = 10;

    NetworkConfig config;
    // will be updated before training
    config.epoch_count = 120;
    config.report_each = 1;
    config.batch_size = 40;
    config.momentum = 0.9f;
    config.learning_rate = 0.001f;

    config.addLayerConfig(dim1, dim2, Activation::Tanh, true, false, false);
    config.addLayerConfig(dim2, dim3, Activation::Tanh, true, false, false);
    config.addLayerConfig(dim3, dim4, Activation::Tanh, true, false, false);
    config.addLayerConfig(dim4, dim5, Activation::Softmax, false, false, false);

    return config;
}
