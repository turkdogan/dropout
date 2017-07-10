#ifndef DROPGRAD_LAYER_H
#define DROPGRAD_LAYER_H

#include "layer.h"

class DropgradLayer : public Layer {

public:
    DropgradLayer(const LayerConfig& layerConfig);

    void feedforward(bool testing = false) override;

    void backpropagate() override;

    void preEpoch(const int epoch) override;

private:
    Eigen::MatrixXf dropout_mask;

    bool first_time;
};

#endif
