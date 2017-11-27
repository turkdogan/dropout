#ifndef DROPGRAD_LAYER_H
#define DROPGRAD_LAYER_H

#include "layer.h"

class DropgradLayer : public Layer {

public:
    DropgradLayer(const LayerConfig& layerConfig);

    void feedforward(bool testing = false) override;

    void backpropagate() override;

    void preEpoch(const int epoch) override;

    void report() override;

private:

    // mask to calculate drop rate for current iteration
    Eigen::MatrixXf m_dropout_mask;

    Eigen::MatrixXf m_avg_dropout_mask;

    Eigen::MatrixXf m_avg_gradient;

    // to calculate the dropout_mean_mask
    int m_current_iteration;

    // int m_total_iteration = 0;

    int m_epoch_count;

    int m_current_epoch;

    double m_avg_keep_rate;
};

#endif
