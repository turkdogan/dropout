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
#include "exp_mnist_dropout.h"
#include "exp_mnist_overfit.h"
#include "exp_cifar.h"
#include "exp_cifar_rates.h"
#include "exp_iris.h"

int main() {
    srand(time(NULL));
    auto first = std::chrono::system_clock::now();

    // MnistDynamicExperiment exp1;
    // exp1.run();

    // MnistRateExperiment exp2;
    // exp2.run();

    // MnistEpochExperiment exp3;
    // exp3.run();

    // MnistActivationExperiment exp4;
    // exp4.run();

    // MnistParamExperiment exp5;
    // exp5.run();

    // MnistDropoutExperiment exp6;
    // exp6.run();

    // IrisExperiment iris_experiment;
    // iris_experiment.run();

    // CifarRateExperiment cifar_experiment;
    // cifar_experiment.run();


    MnistOverfitExperiment exp7;
    exp7.run();

    auto last = std::chrono::system_clock::now();
    auto dur = last - first;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(dur);
    std::cout << secs.count() << " seconds... \n";
    return 0;
}
