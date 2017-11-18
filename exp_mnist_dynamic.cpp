#include "exp_mnist_dynamic.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

void MnistDynamicExperiment::run() {
    std::cout << "Mnist Dropout Dynamic Experiment Run..." << std::endl;
    int total_size = 60000;

    // 60k sample input
    // 10k sample ouput
    Eigen::MatrixXf train_input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
    Eigen::MatrixXf train_output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

    shuffleMatrixPair(train_input, train_output);

    Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
    Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    Scenario s1("NO-DROPOUT");

    int epoch_count = 120;

    auto convex_fn  = [](int epoch) {return epoch * epoch;};
    auto concave_fn = [](int epoch) {return sqrt(epoch);};
    auto linear_fn  = [](int epoch) {return epoch;};

    std::string increasing_key = "INC";
    Scenario s2("L055-095", epoch_count, 0.55f, 0.95f, linear_fn);
    Scenario s3("Concave055-095", epoch_count, 0.55f, 0.95f, concave_fn);
    Scenario s4("Convex0.55-095", epoch_count, 0.55f, 0.95f, convex_fn);

    std::string decreasing_key = "DEC";
    Scenario s2_dec("L095-055", epoch_count, 0.95f, 0.55f, linear_fn);
    Scenario s3_dec("Concave095-055", epoch_count, 0.95f, 0.55f, concave_fn);
    Scenario s4_dec("Convex0.95-055", epoch_count, 0.95f, 0.55f, convex_fn);

    std::string half_key = "HALF";
    Scenario s5("HConcave055-095", epoch_count, epoch_count
                 /4, 0.55f, 0.95f, concave_fn);
    Scenario s6("HConvex0.55-095", epoch_count, epoch_count/4, 0.55f, 0.95f, convex_fn);

    Scenario s7("HConcave095-055", epoch_count, epoch_count/4, 0.95f, 0.55f, concave_fn);
    Scenario s8("HConvex0.95-055", epoch_count, epoch_count/4, 0.95f, 0.55f, convex_fn);

    Scenario scenarios[] = {
        s1,
        s2, s2_dec,
        s3, s3_dec,
        s4, s4_dec,
        s5,
        s6,
        s7,
        s8,
    };

    NetworkConfig config = getConfig();

    for (Scenario scenario : scenarios) {
        config.scenario = scenario;

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
        training_result.category = "Mnist_dynamic";

        std::cout << "write training result... " << std::endl;
        writeTrainingResult(training_result, scenario_name + ".txt", false);
    }

}

NetworkConfig MnistDynamicExperiment::getConfig() {
    const int dim1 = 784;
    const int dim2 = 200;
    const int dim3 = 100;
    const int dim4 = 10;

    NetworkConfig config;
    // will be updated before training
    config.epoch_count = 120;
    config.report_each = 2;
    config.batch_size = 40;
    config.momentum = 0.9f;
    config.learning_rate = 0.01f;
    config.clip_before_error = false;

    config.addLayerConfig(dim1, dim2, Activation::Sigmoid, true, false, false);
    config.addLayerConfig(dim2, dim3, Activation::Sigmoid, true, false, false);
    config.addLayerConfig(dim3, dim4, Activation::Softmax, false, false, false);

    return config;
}
