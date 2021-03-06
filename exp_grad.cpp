#include "exp_grad.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistDropgradExperiment::run() {
    std::cout << "Mnist Dropout Experiment Run..." << std::endl;

    int total_size = 60000;
    int validation_size = 100;
    int validation_begin = total_size - validation_size;

    Eigen::MatrixXf input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(input, output);

    Eigen::MatrixXf validation_input =
        input.block(validation_begin, 0, validation_size, input.cols());
    Eigen::MatrixXf validation_output =
        output.block(validation_begin, 0, validation_size, output.cols());

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int dataset_sizes[] = {59900};
    for (int trial = 0; trial < 1; trial++) {
        for (auto &dataset_size : dataset_sizes) {
            runMnistNetwork(trial,
                            dataset_size,
                            input,
                            output,
                            validation_input,
                            validation_output,
                            test_input,
                            test_output);
        }
    }
}

int MnistDropgradExperiment::reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Eigen::MatrixXf MnistDropgradExperiment::readMnistInput(const std::string& path,
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

Eigen::MatrixXf MnistDropgradExperiment::readMnistOutput(const std::string& path,
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

void MnistDropgradExperiment::runMnistNetwork(int trial,
                     int dataset_size,
                     Eigen::MatrixXf& input,
                     Eigen::MatrixXf& output,
                     Eigen::MatrixXf& validation_input,
                     Eigen::MatrixXf& validation_output,
                     Eigen::MatrixXf& test_input,
                     Eigen::MatrixXf& test_output) {

    NetworkConfig config = getConfig();

    Eigen::MatrixXf train_input = input.block(0, 0, dataset_size, input.cols());
    Eigen::MatrixXf train_output = output.block(0, 0, dataset_size, output.cols());

    Network network(config);
    TrainingResult training_result = network.trainNetwork(
        train_input, train_output,
        validation_input, validation_output,
        false);

    int correct = network.test(test_input, test_output);
    training_result.count = 10000;
    training_result.correct = correct;
    training_result.trial = trial;
    training_result.dataset_size = dataset_size;
    training_result.correct = correct;
    std::string scenario_name =
        std::to_string(dataset_size);
    training_result.name = scenario_name;
    // TODO update category here...
    training_result.category = "DROPGRAD";

    writeTrainingResult(training_result, scenario_name + ".txt");
}

NetworkConfig MnistDropgradExperiment::getConfig() {
    const int dim1 = 784;
    const int dim2 = 200;
    const int dim3 = 120;
    const int dim4 = 40;
    const int dim5 = 10;

    NetworkConfig config;
    config.epoch_count = 120;
    config.report_each = 2;
    config.batch_size = 20;
    config.momentum = 0.9f;
    config.learning_rate = 0.001f;

    config.addLayerConfig(dim1, dim2, Activation::Tanh, false, true);
    config.addLayerConfig(dim2, dim3, Activation::Tanh, false, true);
    config.addLayerConfig(dim3, dim5, Activation::Softmax, false, false);
    /* config.addLayerConfig(dim3, dim3, Activation::Sigmoid, true); */
    // config.addLayerConfig(dim4, dim5, Activation::Softmax, false, false);

    return config;
}
