#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Dense"

#include <vector>
#include <random>
#include <iostream>

void shuffleMatrixPair(Eigen::MatrixXf& mat1, Eigen::MatrixXf& mat2)
{
	if (mat1.rows() != mat2.rows() || mat2.cols() != mat2.cols()) {
		std::cerr << "Not possible to shuffle, dimension problem!" << std::endl;
	}
	int half = static_cast<int>(mat1.rows() * 0.5);
	for (auto i = 0; i < half; i++) {
		int swap_index = rand() % mat1.rows();
		mat1.row(i).swap(mat1.row(swap_index));
		mat2.row(i).swap(mat2.row(swap_index));
	}
}

void splitMatrixPair(const Eigen::MatrixXf& mat1, const Eigen::MatrixXf& mat2,
	std::vector<Eigen::MatrixXf>& mat1_buf,
	std::vector<Eigen::MatrixXf>& mat2_buf,
	int batch_size)
{
	if (mat1.rows() != mat2.rows()) {
		std::cerr << "Not possible to split, dimension problem!" << std::endl;
	}
	mat1_buf.clear();
	mat2_buf.clear();

	int number_of_matrices = mat1.rows() / batch_size;
	int current_row_index = 0;

	for (int i = 0; i < number_of_matrices; i++) {
		mat1_buf.push_back(mat1.block(current_row_index, 0, batch_size, mat1.cols()));
		mat2_buf.push_back(mat2.block(current_row_index, 0, batch_size, mat2.cols()));
		current_row_index += batch_size;
	}
}

void fill_random(Eigen::MatrixXf& mat)
{
	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
			mat(r, c) = -1 + 2 * ((float)rand()) / RAND_MAX;
		}
	}
}

void fill_random_normal(Eigen::MatrixXf& mat,
	float mean = 0.0f, float stddev = 0.1f)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stddev);

	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
			mat(r, c) = distribution(generator);
		}
	}
}

void fill_uniform(Eigen::MatrixXf& mat, float a)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-a, a);

	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
			mat(r, c) = distribution(generator);
		}
	}
}

Eigen::MatrixXf uniformMatrix(int rows, int cols, float low, float high) {
	Eigen::MatrixXf result(rows, cols);

	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(low, high);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			result(r, c) = distribution(generator);
		}
	}
	return result;
}

Eigen::MatrixXf xavierMatrix(int rows, int cols, bool is_sigmoid = true) {
	float scale = is_sigmoid ? 4.0f : 1.0f;
	float high = scale * std::sqrt(6.0f / (rows + cols));

	return uniformMatrix(rows, cols, -high, high);
}

Eigen::MatrixXf binomial(int rows, int cols, double ratio)
{
	std::default_random_engine generator;
	std::binomial_distribution<int> distribution(1, ratio);

	Eigen::MatrixXf result(rows, cols);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			result(r, c) = distribution(generator);
		}
	}
	return result;
}

Eigen::MatrixXf binomial(const Eigen::MatrixXf& mat)
{
	Eigen::MatrixXf result(mat.rows(), mat.cols());
	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
            double rnd = ((double) rand() / (RAND_MAX));
            /* std::cout << rnd << " "; */
            if (mat(r, c) > rnd) {
                result(r, c) = 1.0f;
            } else {
                result(r, c) = 0.0f;
            }
		}
	}
	return result;
}

Eigen::MatrixXf _tanh(Eigen::MatrixXf& mat)
{
	return mat.array().tanh();
}

Eigen::MatrixXf _dtanh(Eigen::MatrixXf& mat)
{
	return 1.0f - mat.array().pow(2.0f);
}

Eigen::MatrixXf sigmoid(Eigen::MatrixXf& mat)
{
	return 1.0f / (1.0f + (-mat.array()).exp());
}

Eigen::MatrixXf dsigmoid(Eigen::MatrixXf& mat)
{
	return mat.array() * (1.0f - mat.array());
}

Eigen::MatrixXf relu(Eigen::MatrixXf& mat)
{
	return (mat.array() > 0).select(mat, 0.0);
}

Eigen::MatrixXf drelu(Eigen::MatrixXf& mat)
{
	Eigen::MatrixXf zeros = Eigen::MatrixXf::Zero(mat.rows(), mat.cols());
	return (mat.array() > 0).select(1.0f, zeros);
}

Eigen::MatrixXf softmax(Eigen::MatrixXf& mat)
{
	Eigen::MatrixXf result(mat.rows(), mat.cols());
	for (auto r = 0; r < mat.rows(); r++) {
		float max = -1.0f * INFINITY;
		for (auto c = 0; c < mat.cols(); c++) {
			if (mat(r, c) > max) {
				max = mat(r, c);
			}
		}
		float sum = 0.0f;
		for (auto c = 0; c < mat.cols(); c++) {
			sum += std::exp(mat(r, c) - max);
		}
		for (auto c = 0; c < mat.cols(); c++) {
			float value = std::exp(mat(r, c) - max) / sum;
			result(r, c) = value;
		}
	}
	return result;
}

Eigen::MatrixXf clipZero(Eigen::MatrixXf& mat)
{
	return (mat.array() <= 0).select(0.000001, mat);
}

Eigen::MatrixXf dsoftmax(Eigen::MatrixXf& mat)
{
	return Eigen::MatrixXf::Ones(mat.rows(), mat.cols());
}

// returns matrix with values between 0 and 1
Eigen::MatrixXf getRandomMatrix(int rows, int cols) {
	return (Eigen::MatrixXf::Random(rows, cols).array() + 1.0f) * 0.5f;
}

#endif
