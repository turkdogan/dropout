#include "layer_grad.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig),
      m_epoch_count(layerConfig.epoch_count),
      m_current_epoch(1),
      m_counter(0),
      m_drop(false) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (!testing) {
        postFeedforward();
    }
}

void DropgradLayer::backpropagate() {
    DZ = D.cwiseProduct(dactivation(Y));
    m_gradient = DZ;
    if (m_drop) {
        DZ = DZ.cwiseProduct(dropout_mask);
    }
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();
}

void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);
    m_current_epoch = epoch;
}

void DropgradLayer::postFeedforward() {
    m_drop = true;
    // technique1();
    // technique2();
    technique3();
}

void DropgradLayer::technique1() {
    if (m_counter > 0) {
        // we care the abstract value of the gradient
        Eigen::MatrixXf abs_gradient = m_gradient.cwiseAbs();
        // calculate average drop rate mask at each iteration
        dropout_avg_mask = (dropout_avg_mask * m_counter + abs_gradient) / (m_counter + 1);
        // keep rate
        // Eigen::MatrixXf keep_rates = 1.0f - abs_gradient.array();
        Eigen::MatrixXf keep_rates = 1.0f - dropout_avg_mask.array();

        double avg_keep_rate = keep_rates.mean();

        dropout_mask = binomial(keep_rates);

        // std::cout << avg_keep_rate << " ";

        Y = Y.cwiseProduct(dropout_mask) * (1.0f / avg_keep_rate);

        // dropout_mask = binomial(Y.rows(), Y.cols(), 0.6f);
        // Y = Y.cwiseProduct(dropout_mask) * (1.0f/0.6f);
    } else {
        dropout_mask = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());
        dropout_avg_mask = dropout_mask;
    }
    m_counter++;
}

void DropgradLayer::technique2() {
}

void DropgradLayer::technique3() {
    if (m_counter > 0) {
        Eigen::MatrixXf abs_gradient = m_gradient.cwiseAbs();
        dropout_mask = binomial(1.0 - abs_gradient.array());
        double avg_keep_rate = dropout_mask.mean();
        // std::cout << dropout_mask.mean() << " ";
        Y = Y.cwiseProduct(dropout_mask) * (1.0f / avg_keep_rate);
    } else {
        dropout_mask = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());
        dropout_avg_mask = dropout_mask;
    }
    m_counter++;
}

