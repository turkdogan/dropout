#include "dropgrad_layer.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (testing) {
        std::cout << "test";
        Y = Y * dropout_avg;
    }
    else {
        if (m_counter > 0) {
            std::cout << "fed ";
            Eigen::MatrixXf abs_gradient = m_gradient.cwiseAbs();
            dropout_avg_mask = (dropout_avg_mask * m_counter + abs_gradient) / (m_counter + 1);
            dropout_mask = binomial(1.0f - dropout_avg_mask.array());
            dropout_avg = (dropout_avg * m_counter + dropout_mask) / (m_counter + 1);
        } else {
            dropout_mask = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());
            dropout_avg_mask = dropout_mask;
            dropout_avg = dropout_mask;
        }
    }
    m_counter++;
}

void DropgradLayer::backpropagate() {
    std::cout << "back ";
    m_gradient = dactivation(Y);
    DZ = D.cwiseProduct(m_gradient).cwiseProduct(dropout_mask);
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
}

void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
}
