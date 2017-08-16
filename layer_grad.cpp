#include "layer_grad.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig),
      m_epoch_count(layerConfig.epoch_count),
      m_current_epoch(0),
      m_current_iteration(1),
      m_drop(false) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (!testing) {
        // in the first iteration of the first epoch
        // we need to set dropout mask by hand
        if (!m_drop) {
            // do not drop for the first epoch
            dropout_mask = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());

            dropout_avg_mask = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());

            m_avg_keep_rate = 1.0;

            m_drop = true;

        } else {
            Y = Y.cwiseProduct(dropout_mask) * (1.0f / m_avg_keep_rate);
        }
    }
}

void DropgradLayer::backpropagate() {
    DZ = D.cwiseProduct(dactivation(Y));

    double alpha = 1.0;

    dropout_avg_mask = (dropout_avg_mask * m_current_iteration + alpha * DZ.cwiseAbs()) /
        (m_current_iteration + 1) ;

    DZ = DZ.cwiseProduct(dropout_mask);
    DW = X.transpose() * DZ;
    DY = DZ * W.transpose();

    // increase iteration after backpropagate phase
    m_current_iteration++;
}

void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);

    m_current_epoch = epoch;

    m_current_iteration = 0;

    if (m_drop) {
        std::cout << "m_drop pre epoch 1" <<std::endl;

        // use previous epoch gradients to calculate current epoch dropout mask
        Eigen::MatrixXf keep_rates = 1.0f - dropout_avg_mask.array();

        m_avg_keep_rate = keep_rates.mean();

        dropout_mask = binomial(keep_rates);

        dropout_avg_mask = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());

        std::cout << "m_drop pre epoch 2" <<std::endl;
    }
}

void DropgradLayer::report() {
    std::cout << "keep rate: " << m_avg_keep_rate<< std::endl;
}
