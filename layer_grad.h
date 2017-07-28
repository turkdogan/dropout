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

    void postFeedforward();

    void technique1();
    void technique2();
    void technique3();

    Eigen::MatrixXf m_gradient;

    // mask to calculate drop rate for current iteration
    Eigen::MatrixXf dropout_mask;

    // average of the previous masks to calcualte current drop rate mask
    Eigen::MatrixXf dropout_avg_mask;

    // average of the applied dropout rates for each neuron
    Eigen::VectorXf dropout_avg;

    bool m_drop;

    // to calculate the dropout_mean_mask
    int m_counter;

    int m_epoch_count;

    int m_current_epoch;
};

#endif
