#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <chrono>

#include "exp_mnist.h"
#include "exp_mnist_epochs.h"
#include "exp_mnist_activations.h"
#include "exp_mnist_params.h"
#include "exp_mnist_rates.h"
#include "exp_mnist_dynamic.h"
#include "exp_cifar.h"
#include "exp_iris.h"

int main() {
    srand(time(NULL));
    auto first = std::chrono::system_clock::now();

    MnistDynamicExperiment experiment;
    // MnistRateExperiment experiment;
    // MnistEpochExperiment experiment;
    // MnistActivationExperiment experiment;
    // MnistParamExperiment experiment;
    // MnistExperiment experiment;
    experiment.run();

    // IrisExperiment iris_experiment;
    // iris_experiment.run();

    auto last = std::chrono::system_clock::now();
    auto dur = last - first;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(dur);
    std::cout << secs.count() << " seconds... \n";
    return 0;
}
