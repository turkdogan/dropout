#include "dropgrad_layer.h"

DropgradLayer::DropgradLayer(const LayerConfig& layerConfig)
    : Layer(layerConfig) {
}

void DropgradLayer::feedforward(bool testing) {
    Layer::feedforward(testing);
    if (testing) {
        // we are multiplying average dropout of each neuron to
        // "compensate for" the dropped neurons during the training phase
        Y = Y.array().rowwise() * dropout_avg.transpose().array();
        std::cout << dropout_avg << std::endl;
    }
    else {
        if (m_counter > 0) {
            // we care the abstract value of the gradient
            Eigen::MatrixXf abs_gradient = m_gradient.cwiseAbs();

            // calculate average drop rate mask at each iteration
            dropout_avg_mask = (dropout_avg_mask * m_counter + abs_gradient) / (m_counter + 1);

            // as dropout average mask decreases over iterations, we should
            // increase the keeping rate graduall over iterations
            Eigen::MatrixXf drop_rates = 1.0f - dropout_avg_mask;

            // apple drop rates
            dropout_mask = binomial(drop_rates);

            // mean of each neuon is calculated over iteration 
            Eigen::VectorXf mean = dropout_mask.colwise().mean();

            dropout_avg = (dropout_avg * m_counter + mean) / (m_counter + 1);
        } else {
            dropout_mask = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());
            dropout_avg_mask = dropout_mask;
            dropout_avg = Eigen::VectorXf::Ones(Y.cols());
        }
    }
    m_counter++;
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
