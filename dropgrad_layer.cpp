#include "dropgrad_layer.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (!testing) {
        if (m_counter > 0) {
            // we care the abstract value of the gradient
            Eigen::MatrixXf abs_gradient = m_gradient.cwiseAbs();

            // calculate average drop rate mask at each iteration
            dropout_avg_mask = (dropout_avg_mask * m_counter + abs_gradient) / (m_counter + 1);

            // keep rate
            // Eigen::MatrixXf keep_rates = 1.0f - abs_gradient.array();
            Eigen::MatrixXf keep_rates = 1.0f - dropout_avg_mask.array();

            double avg_keep_rate = keep_rates.mean();

            // apple drop rates
            dropout_mask = binomial(keep_rates);

            // std::cout << avg_keep_rate << " ";

            Y = Y.cwiseProduct(dropout_mask) * (1.0f / avg_keep_rate);

            // dropout_mask = binomial(Y.rows(), Y.cols(), 0.6f);
            // Y = Y.cwiseProduct(dropout_mask) * (1.0f/0.6f);
        } else {
            dropout_mask = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());
            dropout_avg_mask = dropout_mask;
        }
        m_counter++;
    }
}

void DropgradLayer::backpropagate() {
    m_gradient = dactivation(Y);
    DZ = D.cwiseProduct(m_gradient).cwiseProduct(dropout_mask);
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
}

void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
}
