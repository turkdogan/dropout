#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <chrono>

#include "examples/mnist_experiment.h"
#include "examples/cifar_experiment.h"
#include "examples/iris_experiment.h"


void testConcaveDecScenario() {
    float dropout_begin = 1.0;
    float dropout_end = 0.5;
    int epoch = 100;

    auto fn = [](int epoch){return (double)sqrt(epoch);};
    Scenario scenario("C", epoch, dropout_begin, dropout_end, fn);

    for (int i = 0; i < epoch; i++) {
        // std::cout << fn(i) << ", ";
        std::cout << scenario.getKeepRate(i) << ", ";
    }
    std::cout << std::endl;
}

void testHalfConcaveDecScenario() {

    float dropout_begin = 1.0f;
    float dropout_end = .5f;
    int epoch = 10;

    // auto fn = [](int epoch){return epoch * epoch;};
    auto fn = [](int epoch){return sqrt(epoch);};
    Scenario scenario("HCD", epoch, epoch/2, dropout_begin, dropout_end, fn);

    for (int i = 0; i < epoch; i++) {
        std::cout << scenario.getKeepRate(i) << ", ";
    }
    std::cout << std::endl;
}

void testHalfConvexScenario() {
    auto convex_fn = [](float x) { return x * x; };

    Scenario scenario("foo", 10, 5, 0.5, 1.0, convex_fn);
    for (int i = 0; i < 10; i++) {
        std::cout << scenario.getKeepRate(i) << ", ";
    }
    std::cout << std::endl;
}

void testHalfDecConvexScenario() {
    auto convex_fn = [](float x) { return x * x; };

    Scenario scenario("foo", 10, 5, 1.0, 0.5, convex_fn);
    for (int i = 0; i < 10; i++) {
        std::cout << scenario.getKeepRate(i) << ", ";
    }
    std::cout << std::endl;
}

void testScenarios() {
    // testHalfConvexScenario();
    // testHalfDecConvexScenario();
    // testHalfConcaveDecScenario();
    testConcaveDecScenario();
}

int main() {
	srand(time(NULL));
	auto first = std::chrono::system_clock::now();

    // MnistExperiment mnist_experiment;
    // mnist_experiment.run();

    // CifarExperiment cifar_experiment;
    // cifar_experiment.run();
  testScenarios();

	auto last = std::chrono::system_clock::now();
	auto dur = last - first;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(dur);
	std::cout << secs.count() << " seconds... \n";
	return 0;
}
