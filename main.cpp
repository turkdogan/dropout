#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <chrono>

#include "examples/mnist_experiment.h"
#include "examples/mnist_dropgrad_experiment.h"
#include "examples/cifar_experiment.h"
#include "examples/iris_experiment.h"

void foo() {
    auto mat = Eigen::MatrixXf::Random(10, 10);
    // mat.normalize();
    // std::cout << mat.array() << std::endl;
    // std::cout << mat << std::endl;
}

int main() {
	srand(time(NULL));
	auto first = std::chrono::system_clock::now();

    foo();

    MnistExperiment mnist_experiment;
    mnist_experiment.run();

    // CifarExperiment cifar_experiment;
    // cifar_experiment.run();

	auto last = std::chrono::system_clock::now();
	auto dur = last - first;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(dur);
	std::cout << secs.count() << " seconds... \n";
	return 0;
}
