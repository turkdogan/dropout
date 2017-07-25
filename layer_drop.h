#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"

class DropoutLayer : public Layer {

public:
    DropoutLayer(const LayerConfig& layerConfig, Scenario& scenario)
        : Layer(layerConfig),
        m_scenario(scenario) {
    }

    void feedforward(bool testing = false) override;

    void backpropagate() override;

    void preEpoch(const int epoch) override;

private:
    Eigen::MatrixXf dropout_mask;
    float dropout_ratio = 1.0f;
    std::vector<float> dropouts;

    Scenario m_scenario;
};

#endif
