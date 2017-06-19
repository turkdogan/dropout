#ifndef NETWORK_UTILS_H
#define NETWORK_UTILS_H

#include "network.h"

static void writeTrainingResult(TrainingResult& training_result, std::string file_name) {
    std::ofstream out_file;
    out_file.open("E_" + file_name);

    for (double error : training_result.errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    out_file.open("V_" + file_name);

    for (double error : training_result.validation_errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    float overfit = 0.0f;

    for (int it = 0; it < training_result.errors.size(); it++) {
        overfit += (-training_result.errors[it] + training_result.validation_errors[it]);
    }
    out_file.open("A_" + file_name);
    out_file << training_result.name << ", ";
    out_file << training_result.category << ", ";
    out_file << training_result.trial << ", ";
    out_file << training_result.dataset_size << ", ";
    out_file << training_result.correct<< ", ";
    out_file << overfit << std::endl;
    out_file.close();

    // first laye weights only, not all training_result.weights...
    for (int i = 0; i < 1; i++) {
        std::ofstream w_out_file;
        w_out_file.open("W" + std::to_string(i) + "_" + file_name);
        w_out_file << training_result.weights[i];
        w_out_file.close();
    }
}


#endif
