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
        // calculate the keeping rates after the first gradient calculation
        // therefore for the first feedforward we need to skip this step
        // due to there is not gradient in the first feedforward step
        if (!first_time) {
            // dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
            auto grad = DZ;
            std::cout << grad.rows() << ", " << Y.rows() << std::endl;
            std::cout << grad.cols() << ", " << Y.cols() << std::endl << std::endl;
            // grad.normalize();
            dropout_mask = (grad.array() > 0.5).select(grad, 1.0);
            Y = Y.cwiseProduct(dropout_mask);
        }
    }
}

void DropgradLayer::backpropagate() {
    auto dy = dactivation(Y);
    auto mean = dy.mean();
    if (!first_time) {
        dy = dy.cwiseProduct(dropout_mask);
    } else {
        // DZ will be calculated
        first_time = false;
    }
    DZ = D.cwiseProduct(dy);
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
}

/*
  Set current dropout rate
*/
void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
    first_time = true;
}

