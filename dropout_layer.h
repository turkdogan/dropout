#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"

class DropoutLayer : public Layer {

public:
    DropoutLayer(const LayerConfig& layerConfig, Scenario& scenario)
        : Layer(layerConfig) {
	}

	void feedforward(bool testing = false) override;

	void backpropagate() override;

    void preEpoch(const int epoch) override;

    float averageDropuot() const;

private:
    Eigen::MatrixXf dropout_mask;
	float dropout_ratio = 1.0f;
    std::vector<float> dropouts;

    Scenario m_scenario;
};

#endif
