#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <fstream>
#include <random>

#include "layer.h"
#include "dropout_layer.h"
#include "scenario.h"

struct NetworkConfig {
    // in case of dropout
    Scenario scenario;

    float learning_rate = 0.001f;
    float momentum = 0.9f;

    unsigned int epoch_count = 1;
    unsigned int batch_size = 1;
    unsigned int report_each = 1;

    void addLayerConfig(const int dim1,
                        const int dim2,
                        Activation activation,
                        bool is_dropout = false) {
        LayerConfig layer_config;
        layer_config.rows = dim1;
        layer_config.cols = dim2;
        layer_config.activation = activation;
        layer_config.is_dropout = is_dropout;
        layer_configs.push_back(layer_config);
    }

    std::vector<LayerConfig> layer_configs;
};

struct TrainingResult {
    std::vector<float> errors;
    std::vector<float> validation_errors;

    std::vector<Eigen::MatrixXf> weights;

    int dataset_size;
    int count;
    int correct;
    int trial;

    std::string name;
    std::string category;
};


class Network {

public:
    Network(NetworkConfig& config);

    ~Network();

    TrainingResult trainNetwork(
        Eigen::MatrixXf& input,
        Eigen::MatrixXf& expected,
        Eigen::MatrixXf& v_input,
        Eigen::MatrixXf& v_expected,
        bool skip_validate=true);


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

    Layer **layers;
    int m_layer_count;

    NetworkConfig m_config;
};

#endif
