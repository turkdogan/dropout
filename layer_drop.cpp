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
    DZ = D.cwiseProduct(dy);
    DZ = DZ.cwiseProduct(dropout_mask);
    m_avg_grad += DZ.mean();
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
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
    std::cout << "avg grad: " << m_avg_grad << std::endl;
    m_avg_grad = 0.0;
}
