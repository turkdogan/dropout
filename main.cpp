#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <chrono>

#include "examples/mnist_experiment.h"
#include "examples/mnist_dropgrad_experiment.h"
#include "examples/cifar_experiment.h"
#include "examples/iris_experiment.h"

void foo3() {
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 3);
    Eigen::VectorXf ones = Eigen::VectorXf::Ones(3);
    Eigen::VectorXf mean = mat.colwise().mean();

    std::cout << mean << std::endl;
    std::cout << ones << std::endl;
    ones = ones + mean;
    std::cout << ones << std::endl;
}

void foo2() {
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 3);
    std::cout << mat << std::endl;
    Eigen::VectorXf mean = mat.colwise().mean();
    std::cout << mean << std::endl;

    std::cout << std::endl;
    Eigen::MatrixXf mat2 = Eigen::MatrixXf::Random(4, 3);
    std::cout << mat2 << std::endl << std::endl;
    Eigen::MatrixXf result = mat2.array().rowwise() * mean.transpose().array();
    std::cout << result << std::endl;
    std::cout << std::endl;
    std::cout << result.rows() << "-" << result.cols() << std::endl;

    // auto mat2 = mat.rowwise() * mean;
}

void foo() {
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(1, 1);
    Eigen::MatrixXf mat2 = Eigen::MatrixXf::Random(1, 1) * 2;
    std::cout << mat << std::endl;
    std::cout << mat2 << std::endl << std::endl;
    // mat.normalize();
    // std::cout << mat.array() << std::endl;
    // std::cout << mat << std::endl;

    for (int i = 1; i <= 10; i++) {
        mat = (mat2 + mat * i) / (i + 1);
        std::cout << mat << " ";
        mat2 = mat2 * 0.3f;
    }
    std::cout << mat << std::endl;
}

int main() {
	srand(time(NULL));
	auto first = std::chrono::system_clock::now();

    // MnistExperiment mnist_experiment;
    MnistDropgradExperiment mnist_experiment;
    mnist_experiment.run();

	auto last = std::chrono::system_clock::now();
	auto dur = last - first;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(dur);
	std::cout << secs.count() << " seconds... \n";
	return 0;
}
