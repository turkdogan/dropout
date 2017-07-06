#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <chrono>

#include "examples/mnist_experiment.h"
#include "examples/mnist_dropgrad_experiment.h"
#include "examples/cifar_experiment.h"
#include "examples/iris_experiment.h"

void printScenario() {
    Scenario s3("C0.7", 10, 0.7f);

    for (int i = 0; i < s3.size(); i++) {
        std::cout << s3.getKeepRate(i) << " ";
    }
    s3.print();
    Scenario s4("C0.8", 10, 0.8f);
    s4.print();
}

int main() {
	srand(time(NULL));
	auto first = std::chrono::system_clock::now();

    printScenario();

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
