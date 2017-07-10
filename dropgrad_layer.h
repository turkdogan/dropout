#ifndef DROPGRAD_LAYER_H
#define DROPGRAD_LAYER_H

#include "layer.h"

class DropgradLayer : public Layer {

public:
    DropgradLayer(const LayerConfig& layerConfig);

    void feedforward(bool testing = false) override;

    void backpropagate() override;

    void preEpoch(const int epoch) override;

private:
    Eigen::MatrixXf m_gradient;

    // mask to calculate drop rate for current iteration
    Eigen::MatrixXf dropout_mask;

    // average of the previous masks to calcualte current drop rate mask
    Eigen::MatrixXf dropout_avg_mask;

    // average of the applied dropout rates
    // dropout_mask averages
    Eigen::MatrixXf dropout_avg;

    // to calculate the dropout_mean_mask
    int m_counter;
};

#endif
