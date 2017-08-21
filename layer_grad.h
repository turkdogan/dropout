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
    Eigen::MatrixXf dropout_mask;

    // average of the previous masks to calcualte current drop rate mask
    Eigen::MatrixXf dropout_avg_mask;

    // average of the applied dropout rates for each neuron
    Eigen::VectorXf dropout_avg;

    Eigen::MatrixXf dropout_prev;

    bool m_drop;

    // to calculate the dropout_mean_mask
    int m_current_iteration;

    int m_epoch_count;

    int m_current_epoch;

    double m_avg_keep_rate;
};

#endif
