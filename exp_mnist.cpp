#include "exp_mnist.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

int MnistExperiment::reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Eigen::MatrixXf MnistExperiment::readMnistInput(const std::string& path,
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

Eigen::MatrixXf MnistExperiment::readMnistOutput(const std::string& path,
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

