#ifndef DROPOUT_SCENARIO_H
#define DROPOUT_SCENARIO_H

#include <vector>
#include <functional>

struct DropoutScenario {
    bool dont_drop = false;

    std::vector<double> dropouts;

    double averageDropout() {
        double sum = 0.0f;
        for (double value : dropouts) {
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
    Scenario(std::string name, int epoch_count, double keep_rate);

    // full dropout: use fn to calculate for each epoch
    Scenario(std::string name,
             int epoch_count,
             double keep_begin_rate,
             double keep_end_rate,
             std::function<double(int)>);

    // semi dropout: apply epochs between start_epoch_from -> epoch_count
    Scenario(std::string name,
             int epoch_count,
             int epoch_to_skip,
             double keep_begin_rate,
             double keep_end_rate,
             std::function<double(int)>);

    bool isEnabled() const;

    double getKeepRate(int epoch) const;

    double averageDropout() const;

    int size() const;

    std::string name() const;

private:
    std::vector<double> m_dropouts;

    std::string m_name;
};

#endif
