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
    bool is_dropgrad = false;
};

class Layer {

public:
    Layer(const LayerConfig& layerConfig);

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
