#include <iostream>
#include <map>

#include "network.h"
#include "utils.h"

Network::Network(
    Scenario& scenario,
    NetworkConfig& config,
    LayerConfig& layer_config1,
    LayerConfig& layer_config2,
    LayerConfig& layer_config3)
    : m_scenario(scenario),
      m_config(config),
      l1(layer_config1, scenario),
      l2(layer_config2, scenario),
      l3(layer_config3, createNoDropoutScenario())
{
}

ScenarioResult Network::trainNetwork(
    Eigen::MatrixXf& input,
	Eigen::MatrixXf& target,
	Eigen::MatrixXf& v_input,
	Eigen::MatrixXf& v_target) {

	std::vector<Eigen::MatrixXf> input_buffer;
	std::vector<Eigen::MatrixXf> target_buffer;

	ScenarioResult scenario_result;

	for (int epoch = 1; epoch <= m_config.epoch_count; epoch++) {
        l1.preEpoch(epoch - 1);
        l2.preEpoch(epoch - 1);
        l3.preEpoch(epoch - 1);

		shuffleMatrixPair(input, target);
		splitMatrixPair(input, target, input_buffer, target_buffer, m_config.batch_size);

		float error = 0.0f;
		for (int b = 0; b < input_buffer.size(); b++) {
			error += iterate(input_buffer[b], target_buffer[b]);
		}
		error /= input_buffer.size();
		scenario_result.errors.push_back(error);
		float validation_error = validate(v_input, v_target);
		scenario_result.validation_errors.push_back(validation_error);
		if (epoch % m_config.report_each == 0) {
			std::cout << epoch << ": " << error << ", " << validation_error << std::endl;
		}
	}
    scenario_result.weights.push_back(l1.W);
    scenario_result.weights.push_back(l2.W);
    scenario_result.weights.push_back(l3.W);
	return scenario_result;
}

void Network::feedforward(Eigen::MatrixXf& input, bool testing) {
	Eigen::MatrixXf X = input;
	l1.X = X;
	l1.feedforward(testing);
	l2.X = l1.Y;
	l2.feedforward(testing);
	l3.X = l2.Y;
	l3.feedforward(testing);
}

void Network::backpropagate(Eigen::MatrixXf& error) {
	l3.D = error;
	l3.backpropagate();
	l2.D = l3.DY;
	l2.backpropagate();
	l1.D = l2.DY;
	l1.backpropagate();
}

void Network::update() {
	l3.update(m_config.momentum, m_config.learning_rate);
	l2.update(m_config.momentum, m_config.learning_rate);
	l1.update(m_config.momentum, m_config.learning_rate);
}

float Network::iterate(Eigen::MatrixXf& input, Eigen::MatrixXf& target) {
	feedforward(input, false);
	Eigen::MatrixXf error = l3.Y - target;
	backpropagate(error);
	update();

    // Eigen::MatrixXf clipped = clipZero(l3.Y);
	// Eigen::MatrixXf log = clipped.array().log();
	Eigen::MatrixXf log = l3.Y.array().log();

	return -(target.cwiseProduct(log).sum()) / input.rows();
}

float Network::validate(Eigen::MatrixXf& input, Eigen::MatrixXf& target) {
	feedforward(input, true);
    // auto clipped = clipZero(l3.Y);
	// Eigen::MatrixXf log = clipped.array().log();
	Eigen::MatrixXf log = l3.Y.array().log();
	return -(target.cwiseProduct(log).sum()) / input.rows();
}

int Network::test(Eigen::MatrixXf& input, Eigen::MatrixXf& output) {
	feedforward(input, true);

	auto guessed = l3.Y;

	if (output.cols() > 1) {
		int correct = 0;
		for (int i = 0; i < output.rows(); i++) {
			double max = -1 * INFINITY;
			int max_index = -1;
			for (int j = 0; j < output.cols(); j++) {
				if (guessed(i, j) > max) {
					max = guessed(i, j);
					max_index = j;
				}
			}
			if (output(i, max_index) == 1) {
				correct++;
			}
		}
		std::cout << correct << std::endl;
		return correct;
	}
	else {
		for (int i = 0; i < output.rows(); i++) {
			std::cout << guessed(i, 0) << " - " << output(i, 0) << std::endl;
		}
		return 0;
	}
}

Network::~Network() {
}

