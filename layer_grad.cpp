#include "layer_grad.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig),
      m_epoch_count(layerConfig.epoch_count),
      m_current_epoch(0),
      m_current_iteration(0),
      m_drop(false) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (!testing) {
        // in the first iteration of the first epoch,
        // we need to set dropout mask by hand
        if (!m_drop) {
            dropout_mask = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());
            dropout_avg_mask = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());
            dropout_prev = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());
            m_avg_keep_rate = 1.0;
            m_drop = true;
        } else {
            if (m_current_iteration == 0) {
                Eigen::MatrixXf diff = dropout_prev - dropout_avg_mask;
                Eigen::MatrixXf keep_rates = diff.cwiseAbs();

                //get location of maximum
                Eigen::MatrixXf::Index maxRow, maxCol;
                float max = keep_rates.maxCoeff(&maxRow, &maxCol);
                //get location of minimum
                Eigen::MatrixXf::Index minRow, minCol;
                float min = keep_rates.minCoeff(&minRow, &minCol);

                keep_rates = (diff.array() - min) / (max - min);

                m_avg_keep_rate = keep_rates.mean();
                std::cout << m_avg_keep_rate << std::endl;
                dropout_mask = binomial(keep_rates);

                dropout_prev = dropout_avg_mask;
                dropout_avg_mask = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());
            }
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

}

void DropgradLayer::report() {
    // std::cout << "keep rate: " << m_avg_keep_rate<< std::endl;
}
