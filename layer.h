#ifndef LAYER_H
#define LAYER_H

#include "Eigen/Dense"

#include "utils.h"
#include "scenario.h"

enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax
};

struct LayerConfig {
    int rows;
    int cols;

    Activation activation;
    bool is_dropout = false;
};

class Layer {

public:
    Layer(const LayerConfig& layerConfig) {
        if (layerConfig.activation == Activation::Sigmoid) {
            W = xavierMatrix(layerConfig.rows, layerConfig.cols, true);
        } else if (layerConfig.activation == Activation::Sigmoid) {
            W = xavierMatrix(layerConfig.rows, layerConfig.cols, false);
        } else {
            W = uniformMatrix(layerConfig.rows, layerConfig.cols, -0.05, 0.05);
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

	virtual void feedforward(bool testing = false);

	virtual void backpropagate();

	virtual void update(float momentum, float learning_rate);

    virtual void preEpoch(const int epoch);

public:
	Eigen::MatrixXf X;
	Eigen::MatrixXf Y;

	// to previous layer
	Eigen::MatrixXf DY;

	// from next layer
	Eigen::MatrixXf D;

	Eigen::MatrixXf W;

protected:
	Eigen::MatrixXf W_change;

	Eigen::VectorXf B;
	float B_change;

	Eigen::MatrixXf DW;
	Eigen::MatrixXf DZ;

	Eigen::MatrixXf(*activation)(Eigen::MatrixXf&);
	Eigen::MatrixXf(*dactivation)(Eigen::MatrixXf&);
};

#endif
