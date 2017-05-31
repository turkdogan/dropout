#ifndef XOR_H
#define XOR_H

#include "utils.h"
#include "network.h"

void runXorLayers() {
	std::cout << "Running XOR example..." << std::endl;
	Eigen::MatrixXf input(4, 2);
	input(0, 0) = 0;
	input(0, 1) = 0;
	input(1, 0) = 0;
	input(1, 1) = 1;
	input(2, 0) = 1;
	input(2, 1) = 0;
	input(3, 0) = 1;
	input(3, 1) = 1;

	Eigen::MatrixXf output(4, 1);
	output(0, 0) = 0;
	output(1, 0) = 1;
	output(2, 0) = 1;
	output(3, 0) = 0;

    NetworkConfig config;
    config.epoch_count = 30000;
    config.learning_rate = 0.1f;
    config.momentum = 0.9f;
    config.batch_size = 1;
    config.report_each = 1000;

    Scenario scenario = createNoDropoutScenario();

    LayerConfig layer_config1;
    layer_config1.rows = 2;
    layer_config1.cols = 15;
    layer_config1.activation = Activation::Sigmoid;

    LayerConfig layer_config2;
    layer_config2.rows = 15;
    layer_config2.cols = 5;
    layer_config2.activation = Activation::Sigmoid;

    LayerConfig layer_config3;
    layer_config3.rows = 5;
    layer_config3.cols = 1;
    layer_config3.activation = Activation::Sigmoid;

    Network network(scenario, config, layer_config1, layer_config2, layer_config3);
    ScenarioResult scenario_result = network.trainNetwork(
        input, output,
        input, output);
    int correct = network.test(input, output);
    std::cout << "Correct: " << correct << std::endl;
}


#endif
