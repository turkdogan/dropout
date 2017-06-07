#include "mnist_experiment.h"

#include <iostream>

#include "../utils.h"
#include "../network.h"
#include "../scenario.h"

void MnistExperiment::run() {
    std::cout << "Mnist Dropout Experiment Run..." << std::endl;

	std::ofstream out_file;
	out_file.open("mnist.txt");

    int total_size = 60000;
    int validation_size = 100;
    int validation_begin = total_size - validation_size;

	Eigen::MatrixXf input = readMnistInput("mnist/train-images.idx3-ubyte", total_size);
	Eigen::MatrixXf output = readMnistOutput("mnist/train-labels.idx1-ubyte", total_size);

	Eigen::MatrixXf validation_input =
		input.block(validation_begin, 0, validation_size, input.cols());
	Eigen::MatrixXf validation_output =
		output.block(validation_begin, 0, validation_size, output.cols());

	Eigen::MatrixXf test_input = readMnistInput("mnist/t10k-images.idx3-ubyte", 10000);
	Eigen::MatrixXf test_output = readMnistOutput("mnist/t10k-labels.idx1-ubyte", 10000);

    int dataset_sizes[] = {200, 500, 1000, 2000, 3000};
	for (int trial = 0; trial < 1; trial++) {
        for (auto &dataset_size : dataset_sizes) {

            Eigen::MatrixXf train_input = input.block(0, 0, dataset_size, input.cols());
            Eigen::MatrixXf train_output = output.block(0, 0, dataset_size, output.cols());

            runMnistNetwork(trial,
                            dataset_size,
                            train_input,
                            train_output,
                            validation_input,
                            validation_output,
                            test_input,
                            test_output,
                            out_file);
        }
	}
	out_file.close();
}

int MnistExperiment::reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Eigen::MatrixXf MnistExperiment::readMnistInput(const std::string& path,
                                                       int number_of_items)
{
	std::ifstream file(path, std::ios::binary);
	Eigen::MatrixXf result(number_of_items, 784);

	if (!file.is_open()) {
		std::cerr << "MNIST data file could not be read" << std::endl;
		return result;
	}

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);

	for (int i = 0; i < number_of_items; i++) {
		for (int j = 0; j < n_rows * n_cols; j++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			result(i, j) = (float)temp / 255.0f;
		}
	}
	return result;
}

Eigen::MatrixXf MnistExperiment::readMnistOutput(const std::string& path,
                                                        int number_of_items)
{
	float one_hot_map[10][10] = {
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	};

	Eigen::MatrixXf result(number_of_items, 10);

	std::ifstream file(path, std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "MNIST data file could not be read" << std::endl;
		return result;
	}

	int magic_number = 0;
	int number_of_labels = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_labels, sizeof(number_of_labels));
	number_of_labels = reverseInt(number_of_labels);

	for (int i = 0; i < number_of_items; i++) {
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));
		int index = (int)temp;
		for (int j = 0; j < 10; j++) {
			result(i, j) = one_hot_map[index][j];
		}
	}
	return result;
}

void MnistExperiment::runMnistNetwork(int trial,
                     int dataset_size,
                     Eigen::MatrixXf& train_input,
                     Eigen::MatrixXf& train_output,
                     Eigen::MatrixXf& validation_input,
                     Eigen::MatrixXf& validation_output,
                     Eigen::MatrixXf& test_input,
                     Eigen::MatrixXf& test_output,
                     std::ofstream& out_file) {

	const int dim1 = 784;
	const int dim2 = 200;
	const int dim3 = 100;
	const int dim4 = 10;

	NetworkConfig config;
	config.epoch_count = 120;
	config.report_each = 4;
	config.batch_size = 10;
	config.momentum = 0.9f;
	config.learning_rate = 0.002f;

	config.addLayerConfig(dim1, dim2, Activation::ReLU, true);
	config.addLayerConfig(dim2, dim3, Activation::ReLU, true);
	/* config.addLayerConfig(dim3, dim3, Activation::Sigmoid, true); */
	config.addLayerConfig(dim3, dim4, Activation::Softmax, false);

	Scenario s1 = createNoDropoutScenario();
	Scenario s2 = createConstantDropoutScenario(0.5f, config.epoch_count);
	Scenario s3 = createConstantDropoutScenario(0.7f, config.epoch_count);
	Scenario s4 = createConstantDropoutScenario(0.8f, config.epoch_count);
	Scenario s5 = createConstantDropoutScenario(0.9f, config.epoch_count);
	Scenario s6 = createLinearDropoutScenario(0.55f, 0.95f, config.epoch_count);
	Scenario s7 = createConcaveDropoutScenario(0.55f, 0.95f, config.epoch_count);
	Scenario s8 = createConvexDropoutScenario(0.55f, 0.95f, config.epoch_count);
	Scenario s9 = createConcaveDecDropoutScenario(1.0f, 0.5f, config.epoch_count);

	std::vector<Scenario> scenarios = {s4, s7, s9};
	/* std::vector<Scenario> scenarios = {s1, s2, s3, s4, s5, s6, s7, s8}; */

	for (Scenario& scenario : scenarios) {
		std::cout << "Running: " << scenario.name << std::endl;
		srand(trial + 25);

		Network network(scenario, config);
		ScenarioResult scenario_result = network.trainNetwork(
			train_input, train_output,
			validation_input, validation_output,
			false);

		int correct = network.test(test_input, test_output);
		scenario_result.count = 10000;
		scenario_result.correct = correct;
		scenario_result.trial = trial;
		scenario_result.dataset_size = dataset_size;
		scenario_result.correct = correct;
		std::string scenario_name =
			"MNIST_" +
			std::to_string(dataset_size) + "_" +
			/* std::to_string(trial) + "_" + */
			scenario.name + ".txt";
		scenario_result.scenario_name = scenario_name;
		//writeScenarioResult(scenario_result, scenario_name);

		out_file << scenario_result.scenario_name << ", ";
		out_file << scenario_result.trial << ", ";
		out_file << scenario_result.dataset_size << ", ";
		out_file << scenario_result.correct<< ", ";

		float overfit = 0.0f;

		for (int it = 0; it < scenario_result.errors.size(); it++) {
			overfit += (-scenario_result.errors[it] + scenario_result.validation_errors[it]);
		}
		out_file << overfit << std::endl;
	}
}
