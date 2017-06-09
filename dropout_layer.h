#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"

class DropoutLayer : public Layer {

public:
    DropoutLayer(const LayerConfig& layerConfig, DropoutScenario& scenario)
        : Layer(layerConfig) {
        m_scenario = scenario;
    }

    void feedforward(bool testing = false) override;

    void backpropagate() override;

    void preEpoch(const int epoch) override;

    float averageDropuot() const;

private:
    Eigen::MatrixXf dropout_mask;
    float dropout_ratio = 1.0f;
    std::vector<float> dropouts;

    DropoutScenario m_scenario;
};

#endif
