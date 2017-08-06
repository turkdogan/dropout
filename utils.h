#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Dense"

#include <vector>
#include <random>
#include <iostream>

void shuffleMatrixPair(Eigen::MatrixXf& mat1, Eigen::MatrixXf& mat2);

void splitMatrixPair(const Eigen::MatrixXf& mat1, const Eigen::MatrixXf& mat2,
	std::vector<Eigen::MatrixXf>& mat1_buf,
	std::vector<Eigen::MatrixXf>& mat2_buf,
                     int batch_size);

void fill_random(Eigen::MatrixXf& mat);

void fill_random_normal(Eigen::MatrixXf& mat,
                        float mean = 0.0f,
                        float stddev = 0.1f);

void fill_uniform(Eigen::MatrixXf& mat, float a);

Eigen::MatrixXf uniformMatrix(int rows, int cols, float low, float high);

Eigen::MatrixXf xavierMatrix(int rows, int cols, bool is_sigmoid = true);

Eigen::MatrixXf binomial(int rows, int cols, double ratio);

// keep rate matrix, each element of generated matrix
// will be determined the corresponding value of the mat matrix
// a random value between 0 and 1 is generated, if this
// specific value is less then corresponding element of mat
// then generated matrix element is set to 1 or else 0
Eigen::MatrixXf binomial(const Eigen::MatrixXf& mat);

Eigen::MatrixXf _tanh(Eigen::MatrixXf& mat);

Eigen::MatrixXf _dtanh(Eigen::MatrixXf& mat);

Eigen::MatrixXf sigmoid(Eigen::MatrixXf& mat);

Eigen::MatrixXf dsigmoid(Eigen::MatrixXf& mat);

Eigen::MatrixXf relu(Eigen::MatrixXf& mat);

Eigen::MatrixXf drelu(Eigen::MatrixXf& mat);

Eigen::MatrixXf softmax(Eigen::MatrixXf& mat);

Eigen::MatrixXf clipZero(Eigen::MatrixXf& mat);

Eigen::MatrixXf dsoftmax(Eigen::MatrixXf& mat);

// returns matrix with values between 0 and 1
Eigen::MatrixXf getRandomMatrix(int rows, int cols);

#endif
