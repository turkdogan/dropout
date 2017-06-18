#ifndef DROPOUT_SCENARIO_H
#define DROPOUT_SCENARIO_H

#include <vector>
#include <functional>

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
    Scenario(std::string name);

    // constant dropout
    Scenario(std::string name, int epoch_count, float keep_rate);

    // full dropout: use fn to calculate for each epoch
    Scenario(std::string name,
             int epoch_count,
             float keep_begin_rate,
             float keep_end_rate,
             std::function<int(int)>);

    // semi dropout: apply epochs between start_epoch_from -> epoch_count
    Scenario(std::string name,
             int epoch_count,
             int epoch_to_skip,
             float keep_begin_rate,
             float keep_end_rate,
             std::function<int(int)>);

    bool isEnabled() const;

    float getKeepRate(int epoch) const;

    float averageDropout() const;

    int size() const;

private:
    std::vector<float> m_dropouts;

    std::string m_name;
};

#endif
