#include "layer.h"

void Layer::feedforward(bool testing) {
    Eigen::MatrixXf z1 = (X * W).rowwise() + B.transpose();
    Y = activation(z1);
}

void Layer::backpropagate() {
    DZ = D.cwiseProduct(dactivation(Y));
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
}

void Layer::update(float momentum, float learning_rate) {
    Eigen::MatrixXf w3_change = W_change * momentum + DW * learning_rate;
    float b3_change = B_change * momentum + DZ.mean() * learning_rate;

    W -= w3_change;
    W_change = w3_change;
    B.array() -= b3_change;
    B_change = b3_change;
}

void Layer::preEpoch(const int epoch) {
    // do nothing 
}
