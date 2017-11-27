#include "exp_iris.h"

#include <iostream>

#include "utils.h"
#include "network.h"
#include "scenario.h"

void IrisExperiment::run() {
	NetworkConfig config;
	config.epoch_count = 10000;
	config.learning_rate = 0.1f;
	config.momentum = 0.9f;
	config.batch_size = 20;
	config.report_each = 100;

    Scenario scenario("C0.5", config.epoch_count, 0.5f);
    config.scenario = scenario;

    // Dropout network layer
    config.addLayerConfig(4, 150, Activation::Sigmoid, false, true, false);
    // Standard network layer
    config.addLayerConfig(150, 40, Activation::Sigmoid, false, true, false);
    // Standard network layer
    config.addLayerConfig(40, 3, Activation::Softmax, false, false);

    Network network(config);

	Dataset dataset = readIris();
    if (dataset.input.size() <= 0) {
        std::cerr << "Iris could not be read..." << std::endl;
        return;
    }
	TrainingResult scenario_result = network.trainNetwork(
		dataset.input, dataset.output,
		dataset.input, dataset.output);
    std::cout << "iris runned" << std::endl;
	int correct = network.test(dataset.input, dataset.output);
	std::cout << "Correct: " << correct << std::endl;
}

Dataset IrisExperiment::readIris() {
	std::ifstream file("iris/iris.data", std::ios::in);
	if (!file.is_open()) {
		std::cerr << "Iris data file could not be read" << std::endl;
		return Dataset{};
	}
	std::string str;
	int row = 0;

	std::vector<float> inputs;
	std::vector<float> outputs;

	while (std::getline(file, str)) {
		std::stringstream ss(str);
		float value;
		int col = 0;
		for (std::string s; ss >> value;) {
			if (ss.peek() == ',') {
				ss.ignore();
			}
			if (col == 4) {
				int out_val = (int)(value * 2);
				if (out_val == 0) {
					outputs.push_back(1);
					outputs.push_back(0);
					outputs.push_back(0);
				}
				else if (out_val == 1) {
					outputs.push_back(0);
					outputs.push_back(1);
					outputs.push_back(0);
				}
				else if (out_val == 2) {
					outputs.push_back(0);
					outputs.push_back(0);
					outputs.push_back(1);
				}
			}
			else {
				inputs.push_back(value);
				col++;
			}
		}
		row++;
	}

	int index = 0;
	Eigen::MatrixXf input(150, 4);
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			input(i, j) = inputs[index++];
		}

	}
	index = 0;
	Eigen::MatrixXf output(150, 3);
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 3; j++) {
			output(i, j) = outputs[index++];
		}
	}

	float maxes[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			if (input(i, j) > maxes[j]) {
				maxes[j] = input(i, j);
			}
		}
	}
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			input(i, j) /= maxes[j];
		}
	}
	return Dataset{ input, output };
}
