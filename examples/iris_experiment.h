#ifndef IRIS_EXPERIMENT_H
#define IRIS_EXPERIMENT_H

#include "../experiment.h"

#include "../Eigen/Dense"

struct Dataset {
	Eigen::MatrixXf input;
	Eigen::MatrixXf output;
};

class IrisExperiment : public Experiment {

public:
    void run() override;

private:
    Dataset IrisExperiment::readIris() {
};

#endif
