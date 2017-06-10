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

    std::string scenario_name;
};


class Network {

public:
    Network(
        DropoutScenario& scenario,
        NetworkConfig& config);

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

    DropoutScenario m_scenario;
    NetworkConfig m_config;

};

static void writeTrainingResult(TrainingResult& scenario_result, std::string file_name) {
    std::ofstream out_file;
    out_file.open("E_" + file_name);

    for (double error : scenario_result.errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    out_file.open("V_" + file_name);

    for (double error : scenario_result.validation_errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    float overfit = 0.0f;

    for (int it = 0; it < scenario_result.errors.size(); it++) {
        overfit += (-scenario_result.errors[it] + scenario_result.validation_errors[it]);
    }
    out_file.open("A_" + file_name);
    out_file << scenario_result.scenario_name << ", ";
    out_file << scenario_result.trial << ", ";
    out_file << scenario_result.dataset_size << ", ";
    out_file << scenario_result.correct<< ", ";
    out_file << overfit << std::endl;
    out_file.close();

    // first laye weights only, not all scenario_result.weights...
    for (int i = 0; i < 1; i++) {
        std::ofstream w_out_file;
        w_out_file.open("W" + std::to_string(i) + "_" + file_name);
        w_out_file << scenario_result.weights[i];
        w_out_file.close();
    }
}

#endif
