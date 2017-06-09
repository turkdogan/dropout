#ifndef SCENARIO_H
#define SCENARIO_H

#include <fstream>

#include "Eigen/Dense"
#include <vector>

struct ScenarioResult {
	std::vector<float> errors;
    std::vector<float> validation_errors;

	std::vector<Eigen::MatrixXf> weights;

    int dataset_size;
	int count;
	int correct;
    int trial;

    std::string scenario_name;
};

struct DropoutScenario {
    bool dont_drop = false;

    std::vector<float> dropouts;

    float averageDropout() {
        float sum = 0.0f;
        for (float value : dropouts) {
            sum += value;
        }
        return sum / dropouts.size();
    }

    std::string name;
};

static DropoutScenario createNoDropoutScenario() {
    DropoutScenario scenario;
    scenario.dont_drop = true;
    scenario.name = "NO_DROPOUT";
    return scenario;
}

static DropoutScenario createConstantDropoutScenario(float dropout = 0.5f,
                                   int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "CONSTANT_DROPOUT_" + std::to_string(dropout);

    for (int i = 0; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout);
    }
    return scenario;
}

static DropoutScenario createLinearDropoutScenario(
    float dropout_begin = 0.5f,
    float dropout_end = 1.0f,
    int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "LINEAR_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);

    auto linear_fn = [](float x) { return sqrt(x); };
    float linear_diff = linear_fn(static_cast<float>(epoch_count)) - linear_fn(0.0f);
    float linear_fn_scale = linear_diff / (dropout_end - dropout_begin);
    for (int i = 0; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin + linear_fn(static_cast<float>(i)) / linear_fn_scale);
    }
    return scenario;
}

static DropoutScenario createConcaveDropoutScenario(
                                 float dropout_begin = 0.5f,
                                 float dropout_end = 1.0f,
                                 int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "CONCAVE_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);


    auto concave_fn = [](float x) { return sqrt(x); };
    float concave_diff = concave_fn(static_cast<float>(epoch_count)) - concave_fn(0.0f);
    float convace_fn_scale = concave_diff / (dropout_end - dropout_begin);
    for (int i = 0; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin + concave_fn(static_cast<float>(i)) / convace_fn_scale);
    }
    return scenario;
}

static DropoutScenario createConcaveDecDropoutScenario(
    float dropout_begin = 1.0f,
    float dropout_end = .5f,
    int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "CONCAVE_DEC_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);

    auto concave_fn = [](float x) { return x * x; };
    float concave_diff = concave_fn(static_cast<float>(epoch_count)) - concave_fn(0.0f);
    float convace_fn_scale = concave_diff / (dropout_end - dropout_begin);
    for (int i = 0; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin - concave_fn(static_cast<float>(i)) / convace_fn_scale);
    }
    return scenario;
}

static DropoutScenario createConvexDecDropoutScenario(
    float dropout_begin = 1.0f,
    float dropout_end = .5f,
    int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "CONVEX_DEC_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);

    auto concave_fn = [](float x) { return sqrt(x); };
    float concave_diff = concave_fn(static_cast<float>(epoch_count)) - concave_fn(0.0f);
    float convace_fn_scale = concave_diff / (dropout_end - dropout_begin);
    for (int i = 0; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin - concave_fn(static_cast<float>(i)) / convace_fn_scale);
    }
    return scenario;
}

static DropoutScenario createConvexDropoutScenario(float dropout_begin = 0.5f,
                                            float dropout_end = 1.0f,
                                            int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "CONVEX_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);

    auto convex_fn = [](float x) { return x * x; };
    float convex_diff = convex_fn(static_cast<float>(epoch_count)) - convex_fn(0.0f);
    float convex_fn_scale = convex_diff / (dropout_end - dropout_begin);
    for (int i = 0; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin + convex_fn(static_cast<float>(i)) / convex_fn_scale);
    }
    return scenario;
}

static void writeScenarioResult(ScenarioResult& scenario_result, std::string file_name) {
    std::ofstream out_file;
    out_file.open("E_" + file_name);

    for (double error : scenario_result.errors) {
        out_file << error << std::endl;
    }
    out_file << "correct: " << scenario_result.correct << std::endl;
    out_file.close();

    /* std::ofstream out_file; */
    out_file.open("V_" + file_name);

    for (double error : scenario_result.validation_errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    float overfit = 0.0f;

    for (int it = 0; it < scenario_result.errors.size(); it++) {
        overfit += (-scenario_result.errors[it] + scenario_result.validation_errors[it]);
    }
    out_file.open("A_" + file_name);
    out_file << scenario_result.scenario_name << ", ";
    out_file << scenario_result.trial << ", ";
    out_file << scenario_result.dataset_size << ", ";
    out_file << scenario_result.correct<< ", ";
    out_file << overfit << std::endl;
    out_file.close();

    for (int i = 0; i< scenario_result.weights.size(); i++) {
        std::ofstream w_out_file;
        w_out_file.open("W" + std::to_string(i) + "_" + file_name);
        w_out_file << scenario_result.weights[i];
        w_out_file.close();
    }
}

#endif
