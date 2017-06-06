#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <chrono>

#include "xor.h"
#include "iris.h"
#include "mnist.h"

#include "examples/mnist_dropout_experiment.h"

void selectTest() {
	Eigen::MatrixXf mat = Eigen::MatrixXf::Random(5, 4);
	std::cout << mat << std::endl;
	// mat = (mat < 0).select(0, mat);
	std::cout << "----" << std::endl;
	std::cout << (mat.array() < 0).select(0, mat) << std::endl;
}

int main() {
	srand(time(NULL));
	auto first = std::chrono::system_clock::now();
	// runIris();
	// runXorLayers();
	// runMnist();
	// runCifar();
	// selectTest();

    MnistDropoutExperiment mnist_dropout_ex;
    mnist_dropout_ex.run();

	auto last = std::chrono::system_clock::now();
	auto dur = last - first;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(dur);
	std::cout << secs.count() << " seconds... \n";
	return 0;
}
