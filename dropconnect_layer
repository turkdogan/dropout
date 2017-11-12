#include "layer_drop.h"

void DropoutLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (!testing) {
        dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
        Y = Y.cwiseProduct(dropout_mask) * (1.0f/dropout_ratio);
    }
}

void DropoutLayer::backpropagate() {
    auto dy = dactivation(Y);
    // dy = dy.cwiseProduct(dropout_mask) * (1.0f/dropout_ratio);
    m_gradient = D.cwiseProduct(dy);
    // It does not matter to mask with m_gradient or dy
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
