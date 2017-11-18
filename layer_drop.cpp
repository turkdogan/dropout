#include "layer_drop.h"

void DropoutLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (!testing) {
        dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
        Y = Y.cwiseProduct(dropout_mask);
    } else {
        Y = Y * (1.0 - m_scenario.averageDropout());
    }
}

void DropoutLayer::backpropagate() {
    m_gradient = D.cwiseProduct(dactivation(Y));
    m_gradient = m_gradient.cwiseProduct(dropout_mask);
    DW = X.transpose() * m_gradient;
    DY = m_gradient * W.transpose();
}

/*
  Set current dropout rate
*/
void DropoutLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
    dropout_ratio = m_scenario.getKeepRate(epoch);
    dropouts.push_back(dropout_ratio);
}

void DropoutLayer::report() {
    Layer::report();
}
