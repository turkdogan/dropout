#ifndef DROPOUT_SCENARIO_H
#define DROPOUT_SCENARIO_H

#include <vector>
#include <functional>
#include <string>

class Scenario {

public:
    Scenario();

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

    void setCategory(std::string category);

    bool isEnabled() const;

    double getKeepRate(int epoch) const;

    double averageDropout() const;

    int size() const;

    std::string name() const;

    void print() const;

private:
    std::vector<double> m_dropouts;

    std::string m_name;
};

#endif
