#include "network_utils.h"

void writeTrainingResult(TrainingResult& t_result, std::string file_name, bool plot_overfit) {
    std::string out_folder = "output/";
    std::ofstream out_file;
    out_file.open(out_folder + "E_" + file_name);

    out_file << t_result.correct << std::endl;
    for (double error : t_result.errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    out_file.open(out_folder + "V_" + file_name);

    for (double error : t_result.validation_errors) {
        out_file << error << std::endl;
    }
    out_file.close();

    if (plot_overfit) {
        float overfit = 0.0f;

        std::cout << t_result.errors.size() << std::endl;
        std::cout << t_result.validation_errors.size() << std::endl;
        for (int it = 0; it < t_result.errors.size(); it++) {
            float iter_diff = (-t_result.errors[it] + t_result.validation_errors[it]);
            overfit += iter_diff / (t_result.errors[it] + t_result.validation_errors[it]);
        }

        out_file.open(out_folder + "A_" + file_name);
        out_file << t_result.name << ", ";
        out_file << t_result.category << ", ";
        out_file << t_result.trial << ", ";
        out_file << t_result.dataset_size << ", ";
        out_file << t_result.correct<< ", ";
        out_file << overfit << std::endl;
        out_file.close();
    }

    // first laye weights only, not all t_result.weights...
    for (int i = 0; i < 1; i++) {
        std::ofstream w_out_file;
        w_out_file.open(out_folder + "W" + std::to_string(i) + "_" + file_name);
        w_out_file << t_result.weights[i];
        w_out_file.close();
    }
}
