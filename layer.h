#ifndef LAYER_H
#define LAYER_H

#include "utils.h"
#include "scenario.h"

class Layer {

public:
	Layer(const LayerConfig& layerConfig, Scenario& scenario) : m_scenario(scenario) {
		W = xavierMatrix(layerConfig.rows, layerConfig.cols,
						 layerConfig.activation == Activation::Sigmoid);

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

	void feedforward(bool testing = false);

	void backpropagate();

	void update(float momentum, float learning_rate);

	void preEpoch(const int epoch);

	float averageDropuot() const;

public:
	Eigen::MatrixXf X;
	Eigen::MatrixXf Y;

	// to previous layer
	Eigen::MatrixXf DY;

	// from next layer
	Eigen::MatrixXf D;

	Eigen::MatrixXf W;

	virtual void print() {
		std::cout << "Base Layer: " << W.rows() << ", " << W.cols() << std::endl;
	}

protected:
	Eigen::MatrixXf W_change;

	Eigen::VectorXf B;
	float B_change;

	Eigen::MatrixXf DW;
	Eigen::MatrixXf DZ;

	Eigen::MatrixXf(*activation)(Eigen::MatrixXf&);
	Eigen::MatrixXf(*dactivation)(Eigen::MatrixXf&);

private:
	Scenario& m_scenario;
	Eigen::MatrixXf dropout_mask;
	float dropout_ratio = 1.0f;

	std::vector<float> dropouts;

};

#endif
