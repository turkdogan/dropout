#include "dropout_layer.h"

void DropoutLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (testing) {
        Y = Y * averageDropuot();
    }
    else {
        dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
        Y = Y.cwiseProduct(dropout_mask);
    }
}

void DropoutLayer::backpropagate() {
    Layer::backpropagate();
    DZ = DZ.cwiseProduct(dropout_mask);
}

/*
  Set current dropout rate
*/
void DropoutLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
    dropout_ratio = m_scenario.dropouts[epoch];
    dropouts.push_back(dropout_ratio);
}

float DropoutLayer::averageDropuot() const {
    float total = 0.0f;
    for (float dropout : dropouts) {
        total += dropout;
    }
    return total / dropouts.size();
}
