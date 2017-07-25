#include <iostream>
#include <map>

#include "network.h"
#include "utils.h"

Network::Network(
	NetworkConfig& config)
	: m_config(config)
{
	assert(config.layer_configs.size() > 1);

	m_layer_count = config.layer_configs.size();
	layers = new Layer*[m_layer_count];
	for (int i = 0; i < m_layer_count; i++) {
		LayerConfig& layer_config = config.layer_configs[i];

        if (layer_config.is_dropout && config.scenario.isEnabled()) {
            std::cout << "dropout layer..." << std::endl;
            layers[i] = new DropoutLayer(layer_config, config.scenario);
        } else if (layer_config.is_dropgrad) {
            std::cout << "grad layer..." << std::endl;
            layers[i] = new DropgradLayer(layer_config);
        } else {
            std::cout << "layer..." << std::endl;
			layers[i] = new Layer(layer_config);
        }
	}
}

TrainingResult Network::trainNetwork(
	Eigen::MatrixXf& input,
	Eigen::MatrixXf& target,
	Eigen::MatrixXf& v_input,
	Eigen::MatrixXf& v_target,
	bool skip_validate) {

	std::vector<Eigen::MatrixXf> input_buffer;
	std::vector<Eigen::MatrixXf> target_buffer;

	TrainingResult scenario_result;

	for (int epoch = 1; epoch <= m_config.epoch_count; epoch++) {
		for (int i = 0; i < m_layer_count; i++) {
			Layer *l = layers[i];
			l->preEpoch(epoch -1);
		}

		shuffleMatrixPair(input, target);
		splitMatrixPair(input, target, input_buffer, target_buffer, m_config.batch_size);
		float error = 0.0f;
		for (int b = 0; b < input_buffer.size(); b++) {
			error += iterate(input_buffer[b], target_buffer[b]);
		}
		error /= input_buffer.size();
		scenario_result.errors.push_back(error);
		if (!skip_validate) {
			float validation_error = validate(v_input, v_target);
			scenario_result.validation_errors.push_back(validation_error);
			if (epoch % m_config.report_each == 0) {
				std::cout << epoch << ": " << error << ", " << validation_error << std::endl;
			}
		} else {
			if (epoch % m_config.report_each == 0) {
				std::cout << epoch << ": " << error << std::endl;
			}
		}
	}
	for (int i = 0; i < m_layer_count; i++) {
		Layer *l = layers[i];
		scenario_result.weights.push_back(l->W);
	}
	return scenario_result;
}

void Network::feedforward(Eigen::MatrixXf& input, bool testing) {
	layers[0]->X = input;
	for (int i = 0; i < m_layer_count; i++) {
		Layer *l1 = layers[i];
		l1->feedforward(testing);
		if (i < m_layer_count - 1) {
			Layer *l2 = layers[i+1];
			l2->X = l1->Y;
		}
	}
}

void Network::backpropagate(Eigen::MatrixXf& error) {
	layers[m_layer_count-1]->D = error;
	for (int i = m_layer_count - 1; i >= 0; i--) {
		Layer *l2 = layers[i];
		l2->backpropagate();
		if (i > 0) {
			Layer *l1 = layers[i-1];
			l1->D = l2->DY;
		}
	}
}

void Network::update() {
	for (int i = m_layer_count-1; i >= 0; i--) {
		Layer* l = layers[i];
		l->update(m_config.momentum, m_config.learning_rate);
	}
}

float Network::iterate(Eigen::MatrixXf& input, Eigen::MatrixXf& target) {
	feedforward(input, false);
	Eigen::MatrixXf error = layers[m_layer_count-1]->Y - target;
	backpropagate(error);
	update();
	Eigen::MatrixXf clipped = clipZero(layers[m_layer_count-1]->Y);
	Eigen::MatrixXf log = clipped.array().log();
    // Eigen::MatrixXf log = layers[m_layer_count-1]->Y.array().log();
	return -(target.cwiseProduct(log).sum()) / input.rows();
}

float Network::validate(Eigen::MatrixXf& input, Eigen::MatrixXf& target) {
	feedforward(input, true);
	Eigen::MatrixXf clipped = clipZero(layers[m_layer_count-1]->Y);
	Eigen::MatrixXf log = clipped.array().log();
    // Eigen::MatrixXf log = layers[m_layer_count-1]->Y.array().log();
	return -(target.cwiseProduct(log).sum()) / input.rows();
}

int Network::test(Eigen::MatrixXf& input, Eigen::MatrixXf& output) {
	feedforward(input, true);

	auto guessed = layers[m_layer_count-1]->Y;
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
	for (int i = 0; i < m_layer_count; i++) {
		delete layers[i];
	}
	delete []layers;
}
