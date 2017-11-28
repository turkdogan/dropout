#include "layer_grad.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig),
      m_epoch_count(layerConfig.epoch_count),
      m_current_epoch(0),
      m_current_iteration(0) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);

    if (!testing) {
        if (m_current_iteration > 0) {
            Y = Y.cwiseProduct(m_dropout_mask);
        }
    } else {
        float avg_dropout_mask = m_avg_dropout_mask.mean();
        Y = Y * (1.0 - avg_dropout_mask);
    }
}

void DropgradLayer::backpropagate() {
    m_gradient = D.cwiseProduct(dactivation(Y));

    Eigen::MatrixXf abs_gradient = m_gradient.cwiseAbs();

    if (m_current_iteration > 0) {
        m_gradient = m_gradient.cwiseProduct(m_dropout_mask);
    }
    DW = X.transpose() * m_gradient;
    DY = m_gradient * W.transpose();

    if (m_current_iteration == 0) {
        m_avg_gradient = Eigen::MatrixXf(abs_gradient.rows(), abs_gradient.cols());
        m_dropout_mask = Eigen::MatrixXf(abs_gradient.rows(), abs_gradient.cols());
        m_avg_dropout_mask = Eigen::MatrixXf(abs_gradient.rows(), abs_gradient.cols());
    }

    float diff_hyper_param = 0.05;

	for (auto r = 0; r < abs_gradient.rows(); r++) {
		for (auto c = 0; c < abs_gradient.cols(); c++) {
            float diff_abs = abs_gradient(r,c) - m_avg_gradient(r,c);
            float keep_rate = 1.0;
            if (diff_abs < diff_hyper_param) {
                keep_rate = 0.5;
            } else {
                keep_rate = 0.2;
            }
            double rnd = ((double) rand() / (RAND_MAX));
            if (rnd <= keep_rate) {
                m_dropout_mask(r,c) = 1.0;
            } else {
                m_dropout_mask(r,c) = 0.0;
            }
		}
    }

    m_avg_dropout_mask = (m_avg_dropout_mask * m_current_iteration + m_dropout_mask) / (m_current_iteration + 1.0);

    m_avg_gradient = (m_avg_gradient * m_current_iteration + abs_gradient) / (m_current_iteration + 1.0);

    m_current_iteration++;
}

void DropgradLayer::preEpoch(const int epoch) {
    Layer::preEpoch(epoch);

    m_current_epoch = epoch;

    // m_current_iteration = 0;
}

void DropgradLayer::report() {
    // std::cout << "keep rate: " << m_avg_keep_rate<< std::endl;
}
