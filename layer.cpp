#include "layer.h"

Layer::Layer(const LayerConfig& layerConfig) {
    if (layerConfig.activation == Activation::Sigmoid) {
        W = xavierMatrix(layerConfig.rows, layerConfig.cols, true);
    } else if (layerConfig.activation == Activation::Tanh) {
        W = xavierMatrix(layerConfig.rows, layerConfig.cols, false);
    } else {
        int r = layerConfig.rows;
        int c = layerConfig.cols;
        // http://cs231n.github.io/neural-networks-2/
        W = Eigen::MatrixXf::Random(r, c) * sqrt(2.0f/(r + c));
    }
    W_change = Eigen::MatrixXf::Zero(W.rows(), W.cols());
    B = Eigen::VectorXf::Ones(W.cols());
    B_change = 0.0f;

    switch (layerConfig.activation) {
    case Activation::Sigmoid:
        activation = sigmoid;
        dactivation = dsigmoid;
        break;
    case Activation::Tanh:
        activation = _tanh;
        dactivation = _dtanh;
        break;
    case Activation::ReLU:
        activation = relu;
        dactivation = drelu;
        break;
    case Activation::Softmax:
        activation = softmax;
        dactivation = dsoftmax;
        break;
    default:
        break;
    };
}

Layer::~Layer() {
}

void Layer::feedforward(bool testing) {
    Eigen::MatrixXf z1 = (X * W).rowwise() + B.transpose();
    Y = activation(z1);
}

void Layer::backpropagate() {
    m_gradient = D.cwiseProduct(dactivation(Y));
    DW = X.transpose() * m_gradient;
    DY = m_gradient * W.transpose();
}

void Layer::update(float momentum, float learning_rate) {
    Eigen::MatrixXf w_change = W_change * momentum + DW * learning_rate;
    float b_change = B_change * momentum + m_gradient.mean() * learning_rate;

    W -= w_change;
    W_change = w_change;
    B.array() -= b_change;
    B_change = b_change;
}

void Layer::preEpoch(const int epoch) {
    // do nothing 
}

void Layer::report() {
    // do nothing 
}
