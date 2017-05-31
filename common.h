#ifndef COMMON_H
#define COMMON_H

#include "Eigen/Dense"

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
};

#endif
