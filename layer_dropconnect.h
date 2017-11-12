#ifndef DROPCONNECT_LAYER_H
#define DROPCONNECT_LAYER_H

#include "layer.h"

class DropconnectLayer : public Layer {

public:
    DropconnectLayer (const LayerConfig& layerConfig, Scenario& scenario)
        : Layer(layerConfig),
        m_scenario(scenario) {
    }

    void feedforward(bool testing = false) override;

    void backpropagate() override;

    void preEpoch(const int epoch) override;

    void report() override;

private:
    Eigen::MatrixXf m_dropconnect_mask;

    Eigen::MatrixXf m_dropped_weights;

    float dropout_ratio = 1.0f;
    std::vector<float> dropouts;

    Scenario m_scenario;
};

#endif
