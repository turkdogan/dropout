#include "layer_dropconnect.h"

void DropconnectLayer::feedforward(bool testing) {
    if (!testing) {
        m_dropconnect_mask = binomial(W.rows(), W.cols(), dropout_ratio);
        m_dropped_weights = W.cwiseProduct(m_dropconnect_mask) * (1.0f/dropout_ratio);
        Eigen::MatrixXf z1 = (X * m_dropped_weights).rowwise() + B.transpose();
        Y = activation(z1);
    } else {
        Layer::feedforward(testing);
    }
}

void DropconnectLayer::backpropagate() {
    auto dy = dactivation(Y);
    m_gradient = D.cwiseProduct(dy);
    DW = X.transpose() * m_gradient;
    DY = m_gradient * m_dropped_weights.transpose();
}

/*
  Set current dropout rate
*/
void DropconnectLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
    dropout_ratio = m_scenario.getKeepRate(epoch);
    dropouts.push_back(dropout_ratio);
}

void DropconnectLayer::report() {
    Layer::report();
}
