#ifndef DROPOUT_SCENARIO_H
#define DROPOUT_SCENARIO_H

#include <vector>

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

class Scenario {

public:
    // no dropout
    Scenario(std::string name, int epoch_count);
    // constant dropout
    Scenario(std::string name, int epoch_count, float keep_rate);
    // full dropout: use fn to calculate for each epoch
    Scenario(std::string name, int epoch_count, std::function<float>(float));
    // semi dropout: apply epochs between start_epoch_from -> epoch_count
    Scenario(std::string name, int epoch_count, std::function<float>(float), int start_epoch_from);

    bool enabled();

    float getKeepRate(int epoch);
    float averageDropout();

private:
    std::vector<float> m_dropouts;

    std::string m_name;
};

#endif
