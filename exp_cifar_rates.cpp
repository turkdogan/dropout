#include "exp_cifar_rates.h"

#include <iostream>
#include <vector>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

const int DIM_CIFAR = 3072;

void CifarRateExperiment::run() {
    std::cout << "Cifar Dropout Rates Experiment Run..." << std::endl;

    int size_for_bin = 10000;
    int test_size = 10000;

	Eigen::MatrixXf input(size_for_bin * 5, DIM_CIFAR);
	Eigen::MatrixXf label(size_for_bin * 5, 10);

    std::vector<std::string> file_names;
    for (int i = 1; i <= 5; i++) {
        std::string file_name = "cifar-10-batches-bin/data_batch_" + std::to_string(i) +".bin";
        file_names.push_back(file_name);
    }

	readCifarInput(file_names, input, label);

    file_names.clear();
    std::string file_name = "cifar-10-batches-bin/test_batch.bin";
    file_names.push_back(file_name);
	Eigen::MatrixXf test_input(test_size, DIM_CIFAR);
	Eigen::MatrixXf test_label(test_size, 10);
	readCifarInput(file_names, test_input, test_label);

    int dataset_sizes[] = {10000};
    for (int trial = 0; trial < 1; trial++) {
        for (auto &dataset_size : dataset_sizes) {
            runCifar(trial, dataset_size,
                            input,
                            label,
                            test_input,
                            test_label);
        }
    }
}

void CifarRateExperiment::readCifarInput(
    std::vector<std::string> &file_names,
    Eigen::MatrixXf& input_buffer,
    Eigen::MatrixXf& label_buffer)
{

	float map[10][10] = {
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

	unsigned char buffer[DIM_CIFAR];
	unsigned char label;

    int starting_index = 0;
    for (std::string &file_name : file_names) {

        std::ifstream file(file_name, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "CIFAR data file could not be read: " << file_name << std::endl;
            return;
        }

        for (int i = 0; i < 10000; i++) {
            file.read((char*)&label, sizeof(label));
            for (unsigned int j = 0; j < 10; j++) {
                label_buffer(starting_index + i, j) = map[label][j];
            }
            file.read((char*)&buffer, DIM_CIFAR);
            for (unsigned int j = 0; j < DIM_CIFAR; j++) {
                input_buffer(starting_index + i, j) = buffer[j] / 255.0f;
            }
        }
        file.close();
        starting_index += 10000;
    }

}


void CifarRateExperiment::runCifar(int trial,
                                      int dataset_size,
                                      Eigen::MatrixXf& input,
                                      Eigen::MatrixXf& output,
                                      Eigen::MatrixXf& test_input,
                                      Eigen::MatrixXf& test_output) {

    NetworkConfig config = getConfig();

    std::map<std::string, std::vector<Scenario>>&& map = getScenarios(config.epoch_count);

    for (const auto& pair : map) {
        const std::string& category = pair.first;
        std::vector<Scenario> scenarios = pair.second;

        for (Scenario& scenario : scenarios) {
            std::cout << "Running: " << scenario.name() << std::endl;
            srand(trial + 15);

            // read data from scratch
            Eigen::MatrixXf train_input = input.block(0, 0, dataset_size, input.cols());
            Eigen::MatrixXf train_output = output.block(0, 0, dataset_size, output.cols());

            config.scenario = scenario;
            Network network(config);
            TrainingResult training_result = network.trainNetwork(
                train_input, train_output
                );

            int correct = network.test(train_input, train_output);
            training_result.count = 10000;
            training_result.correct = correct;
            training_result.trial = trial;
            training_result.dataset_size = dataset_size;
            training_result.correct = correct;
            std::string scenario_name =
                "CIFAR_" +
                std::to_string(dataset_size) + "_" +
                /* std::to_string(trial) + "_" + */
                scenario.name();
            training_result.name = scenario_name;
            training_result.category = category;
            std::cout << "Writing training result..." << std::endl;
            writeTrainingResult(training_result, scenario_name + ".txt", false);
            std::cout << "written..."<< std::endl;
        }
    }
}


std::map<std::string, std::vector<Scenario>> CifarRateExperiment::getScenarios(int epoch_count) {

    std::map<std::string, std::vector<Scenario>> scenario_map;

    Scenario s1("NO-DROPOUT");
    std::string no_key = "NO";
    scenario_map[no_key].push_back(s1);

    std::string constant_key = "CONSTANT";
    Scenario s2("C0.5", epoch_count, 0.5f);
    scenario_map[constant_key].push_back(s2);

    return scenario_map;
}

NetworkConfig CifarRateExperiment::getConfig() {

	const int dim1 = DIM_CIFAR;
	const int dim2 = 200;
	const int dim3 = 100;
	const int dim4 = 10;

	NetworkConfig config;
	config.epoch_count = 600;
	config.report_each = 2;
	config.batch_size = 100;
	config.momentum = 0.9f;
	config.learning_rate = 0.001f;

	config.addLayerConfig(dim1, dim2, Activation::Sigmoid, true);
	config.addLayerConfig(dim2, dim3, Activation::Sigmoid, true);
	config.addLayerConfig(dim3, dim4, Activation::Softmax, false);

    return config;
}
