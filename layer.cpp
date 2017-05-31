#include "layer.h"

void Layer::feedforward(bool testing) {
    Eigen::MatrixXf z1 = (X * W).rowwise() + B.transpose();
    Y = activation(z1);

    if (m_scenario.type != DropoutType::NONE) {
        if (testing) {
            Y = Y * averageDropuot();
        }
        else {
            dropout_mask = binomial(Y.rows(), Y.cols(), dropout_ratio);
            Y = Y.cwiseProduct(dropout_mask);
        }
    }
}

void Layer::backpropagate() {
    DZ = D.cwiseProduct(dactivation(Y));
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();

    if (m_scenario.type != DropoutType::NONE) {
        DZ = DZ.cwiseProduct(dropout_mask);
    }
}

void Layer::update(float momentum, float learning_rate) {
    Eigen::MatrixXf w3_change = W_change * momentum + DW * learning_rate;
    float b3_change = B_change * momentum + DZ.mean() * learning_rate;

    W -= w3_change;
    W_change = w3_change;
    B.array() -= b3_change;
    B_change = b3_change;
}

/*
  Set current dropout rate
*/
void Layer::preEpoch(const int epoch) {
    if (m_scenario.type != DropoutType::NONE) {
        dropout_ratio = m_scenario.dropouts[epoch];
        dropouts.push_back(dropout_ratio);
    }
}

float Layer::averageDropuot() const {
    float total = 0.0f;
    for (float dropout : dropouts) {
        total += dropout;
    }
    return total / dropouts.size();
}
