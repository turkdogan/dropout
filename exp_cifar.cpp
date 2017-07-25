#include "exp_cifar.h"

#include <iostream>

#include "utils.h"
#include "network_utils.h"
#include "scenario.h"

const int DIM_CIFAR = 3072;

void CifarExperiment::run() {
    std::cout << "Cifar Dropout Experiment Run..." << std::endl;

    int total_size = 10000;
    int validation_size = 100;
    int validation_begin = total_size - validation_size;

	Eigen::MatrixXf input(total_size, DIM_CIFAR);
	Eigen::MatrixXf label(total_size, 10);

	readCifarInput("cifar-10-batches-bin/data_batch_1.bin", input, label, total_size);

    Eigen::MatrixXf validation_input =
        input.block(validation_begin, 0, validation_size, input.cols());
    Eigen::MatrixXf validation_label =
        label.block(validation_begin, 0, validation_size, label.cols());

	Eigen::MatrixXf test_input(total_size, DIM_CIFAR);
	Eigen::MatrixXf test_label(total_size, 10);
	readCifarInput("cifar-10-batches-bin/data_batch_1.bin", test_input, test_label, total_size);

    int dataset_sizes[] = {5000};
    for (int trial = 0; trial < 1; trial++) {
        for (auto &dataset_size : dataset_sizes) {
            runCifar(trial, dataset_size,
                            input,
                            label,
                            validation_input,
                            validation_label,
                            test_input,
                            test_label);
        }
    }
}

void CifarExperiment::readCifarInput(
	const std::string& path,
	Eigen::MatrixXf& input_buffer,
	Eigen::MatrixXf& label_buffer,
	int number_of_items,
	int starting_index)
{
	std::ifstream file(path, std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "CIFAR data file could not be read" << std::endl;
		return;
	}

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

	for (int i = 0; i < number_of_items; i++) {
		file.read((char*)&label, sizeof(label));
		for (unsigned int j = 0; j < 10; j++) {
			label_buffer(starting_index + i, j) = map[label][j];
		}
		file.read((char*)&buffer, DIM_CIFAR);
		for (unsigned int j = 0; j < DIM_CIFAR; j++) {
			input_buffer(starting_index + i, j) = buffer[j] / 255.0f;
		}
	}
}


void CifarExperiment::runCifar(int trial,
                                      int dataset_size,
                                      Eigen::MatrixXf& input,
                                      Eigen::MatrixXf& output,
                                      Eigen::MatrixXf& validation_input,
                                      Eigen::MatrixXf& validation_output,
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
                train_input, train_output,
                validation_input, validation_output, false);

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
            writeTrainingResult(training_result, scenario_name + ".txt");
        }
    }
}


std::map<std::string, std::vector<Scenario>> CifarExperiment::getScenarios(int epoch_count) {

    std::map<std::string, std::vector<Scenario>> scenario_map;

    Scenario s1("NO-DROPOUT");
    std::string no_key = "NO";
    scenario_map[no_key].push_back(s1);

    auto convex_fn  = [](int epoch) {return epoch * epoch;};
    auto concave_fn = [](int epoch) {return sqrt(epoch);};
    auto linear_fn  = [](int epoch) {return epoch;};

    std::string constant_key = "CONSTANT";
    Scenario s2("C0.5", epoch_count, 0.5f);
    Scenario s3("C0.7", epoch_count, 0.7f);
    Scenario s4("C0.8", epoch_count, 0.8f);
    Scenario s5("C0.9", epoch_count, 0.9f);
    scenario_map[constant_key].push_back(s2);
    // scenario_map[constant_key].push_back(s3);
    // scenario_map[constant_key].push_back(s4);
    // scenario_map[constant_key].push_back(s5);

    std::string increasing_key = "INC";
    Scenario s6("L055-095", epoch_count, 0.55f, 0.95f, linear_fn);
    Scenario s6_dec("L095-055", epoch_count, 0.95f, 0.55f, linear_fn);
    Scenario s7("Concave055-095", epoch_count, 0.55f, 0.95f, concave_fn);
    Scenario s8("Convex0.55-095", epoch_count, 0.55f, 0.95f, convex_fn);
    scenario_map[increasing_key].push_back(s6);
    // scenario_map[increasing_key].push_back(s7);
    // scenario_map[increasing_key].push_back(s8);

    std::string decreasing_key = "DEC";
    Scenario s9("Concave095-055", epoch_count, 0.95f, 0.55f, concave_fn);
    Scenario s10("Convex0.95-055", epoch_count, 0.95f, 0.55f, convex_fn);
    // scenario_map[decreasing_key].push_back(s9);
    // scenario_map[decreasing_key].push_back(s10);
    // scenario_map[decreasing_key].push_back(s6_dec);

    std::string half_key = "HALF";
    Scenario s11("HConcave055-095", epoch_count, epoch_count
                 /4, 0.55f, 0.95f, concave_fn);
    Scenario s12("HConvex0.55-095", epoch_count, epoch_count/4, 0.55f, 0.95f, convex_fn);

    Scenario s13("HConcave095-055", epoch_count, epoch_count/4, 0.95f, 0.55f, concave_fn);
    Scenario s14("HConvex0.95-055", epoch_count, epoch_count/4, 0.95f, 0.55f, convex_fn);

    // scenario_map[half_key].push_back(s11);
    // scenario_map[half_key].push_back(s12);
    // scenario_map[half_key].push_back(s13);
    // scenario_map[half_key].push_back(s14);

    return scenario_map;
}

NetworkConfig CifarExperiment::getConfig() {

	const int dim1 = DIM_CIFAR;
	const int dim2 = 200;
	const int dim3 = 100;
	const int dim4 = 10;

	NetworkConfig config;
	config.epoch_count = 320;
	config.report_each = 4;
	config.batch_size = 100;
	config.momentum = 0.9f;
	config.learning_rate = 0.001f;

	config.addLayerConfig(dim1, dim2, Activation::ReLU, true);
	config.addLayerConfig(dim2, dim3, Activation::Sigmoid, true);
	config.addLayerConfig(dim3, dim4, Activation::Softmax, false);

    return config;
}
