#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <fstream>
#include <random>

#include "layer.h"
#include "scenario.h"

struct NetworkConfig {
	int epoch_count = 1;
	float learning_rate = 0.001f;
	float momentum = 0.9f;
	int batch_size = 1;
	int report_each = 1;
};

class Network {

public:
    Network(
        Scenario& scenario,
        NetworkConfig& config,
        LayerConfig& layer_config1,
        LayerConfig& layer_config2,
        LayerConfig& layer_config3);

    ~Network();

    ScenarioResult trainNetwork(
        Eigen::MatrixXf& input,
        Eigen::MatrixXf& expected,
        Eigen::MatrixXf& v_input,
        Eigen::MatrixXf& v_expected);


    int test(
        Eigen::MatrixXf& input,
        Eigen::MatrixXf& output);

private:
    void feedforward(
        Eigen::MatrixXf & input,
        bool testing = false);

    float iterate(
        Eigen::MatrixXf& input,
        Eigen::MatrixXf& output);

    float validate(
        Eigen::MatrixXf& input,
        Eigen::MatrixXf& output);

    void Network::backpropagate(
        Eigen::MatrixXf& error);

    void Network::update();

    Layer l1;
    Layer l2;
    Layer l3;

    Scenario m_scenario;
    NetworkConfig m_config;

};

#endif
