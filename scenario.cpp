#include "scenario.h"

#include <cassert>

Scenario::Scenario(std::string name)
    :m_name (name) {
}

Scenario::Scenario(std::string name, int epoch_count, double keep_rate)
    :m_name(name) {
    m_dropouts.assign(epoch_count, keep_rate);
}

Scenario::Scenario(std::string name,
                   int epoch_count,
                   double keep_begin_rate,
                   double keep_end_rate,
                   std::function<double(int)> generator)
    :m_name(name) {

    bool incremental = keep_end_rate > keep_begin_rate;

    double diff = generator(epoch_count-1);
    double keep_diff = keep_end_rate - keep_begin_rate;
    if (!incremental) {
        keep_diff *= -1;
    }
    double scale = diff / keep_diff;
    m_dropouts.push_back(keep_begin_rate);
    for (int i = 1; i < epoch_count; i++) {
        double generated = generator(i);
        if (incremental) {
            m_dropouts.push_back(keep_begin_rate + generated/scale);
        } else {
            m_dropouts.push_back(keep_begin_rate - generated/scale);
        }
    }
}

Scenario::Scenario(std::string name,
                   int epoch_count,
                   int epoch_to_skip,
                   double keep_begin_rate,
                   double keep_end_rate,
                   std::function<double(int)> generator)
    :m_name(name) {
    m_dropouts.assign(epoch_to_skip, 1.0);

    // generate for epoch_count - epoch_to_skip epochs
    // epoch_to_skip'th rate is keep_begin_rate
    // (epoch_count-1)'th rate is keep_end_rate

    bool incremental = keep_end_rate > keep_begin_rate;

    int n_epoch_to_generate = epoch_count - epoch_to_skip -1;
    double diff = generator(n_epoch_to_generate);
    double keep_diff = keep_end_rate - keep_begin_rate;
    if (!incremental) {
        keep_diff *= -1;
    }
    double scale = diff / (keep_diff);

    m_dropouts.push_back(keep_begin_rate);
    for (int i = epoch_to_skip+1; i < epoch_count; i++) {
        if (incremental) {
            m_dropouts.push_back(keep_begin_rate + generator(i-epoch_to_skip)/scale);
        } else {
            m_dropouts.push_back(keep_begin_rate - generator(i-epoch_to_skip)/scale);
        }
    }
}

double Scenario::getKeepRate(int epoch) const {
    assert(epoch < m_dropouts.size());
    return m_dropouts[epoch];
}

double Scenario::averageDropout() const {
    double sum = 0.0f;
    for (double value : m_dropouts) {
        sum += value;
    }
    return sum / m_dropouts.size();
}

bool Scenario::isEnabled() const {
    return m_dropouts.size() > 0;
}

int Scenario::size() const {
    return m_dropouts.size();
}

std::string Scenario::name() const {
    return m_name;
}
