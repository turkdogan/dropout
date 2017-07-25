#ifndef IRIS_EXPERIMENT_H
#define IRIS_EXPERIMENT_H

#include "exp.h"

#include "Eigen/Dense"

struct Dataset {
	Eigen::MatrixXf input;
	Eigen::MatrixXf output;
};

class IrisExperiment : public Experiment {

public:
    void run() override;

private:
    Dataset readIris();
};

#endif
