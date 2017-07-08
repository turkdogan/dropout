#include "dropgrad_layer.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (testing) {
        // Y = Y * m_scenario.averageDropout();
    }
    else {
        // dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
        auto grad = DZ;
        // grad.normalize();
        dropout_mask = (grad.array() > 0.5).select(grad, 1.0);
        Y = Y.cwiseProduct(dropout_mask);
    }
}

void DropgradLayer::backpropagate() {
    auto dy = dactivation(Y);
    auto mean = dy.mean();
    std::cout << mean << " ";
    dy = dy.cwiseProduct(dropout_mask);
    DZ = D.cwiseProduct(dy);
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
}

/*
  Set current dropout rate
*/
void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
}

