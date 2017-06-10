#ifndef SCENARIO_H
#define SCENARIO_H

#include <fstream>
#include <vector>

#include "Eigen/Dense"

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
    float convace_fn_scale = concave_diff / (dropout_begin - dropout_end);
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
    float convace_fn_scale = concave_diff / (dropout_begin - dropout_end);
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

static DropoutScenario halfConvexDropoutScenario(float dropout_begin = 0.5f,
                                                   float dropout_end = 1.0f,
                                                   int epoch_count = 0) {
    DropoutScenario scenario;
    scenario.name = "HALF_CONVEX_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);


    for (int i = 0; i < epoch_count/2; i++) {
        scenario.dropouts.push_back(1.0f);
    }

    auto convex_fn = [](float x) { return x * x; };
    float convex_diff = convex_fn(static_cast<float>(epoch_count/2)) - convex_fn(0.0f);
    float convex_fn_scale = convex_diff / (dropout_end - dropout_begin);
    for (int i = epoch_count/2; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin + convex_fn(i + 1 - epoch_count/2) / convex_fn_scale);
    }
    return scenario;
}

static DropoutScenario halfConcaveDropoutScenario(
    float dropout_begin = 0.5f,
    float dropout_end = 1.0f,
    int epoch_count = 0) {

    DropoutScenario scenario;
    scenario.name = "HALF_CONCAVE_DROPOUT_" +
        std::to_string(dropout_begin) +
        "_" + std::to_string(dropout_end);

    for (int i = 0; i < epoch_count/2; i++) {
        scenario.dropouts.push_back(1.0f);
    }

    auto concave_fn = [](float x) { return sqrt(x); };
    float concave_diff = concave_fn(static_cast<float>(epoch_count - epoch_count/2)) - concave_fn(0.0f);
    float convace_fn_scale = concave_diff / (dropout_end - dropout_begin);
    for (int i = epoch_count/2; i < epoch_count; i++) {
        scenario.dropouts.push_back(dropout_begin + concave_fn(i + 1 - epoch_count/2) / convace_fn_scale);
    }
    return scenario;
}

#endif
