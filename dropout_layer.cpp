#include "dropout_layer.h"

void DropoutLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (testing) {
        Y = Y * m_scenario.averageDropout();
    }
    else {
        dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
        Y = Y.cwiseProduct(dropout_mask);
    }
}

void DropoutLayer::backpropagate() {
    DZ = D.cwiseProduct(dactivation(Y));
    DZ = DZ.cwiseProduct(dropout_mask);
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
