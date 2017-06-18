#include "dropout_scenario.h"

#include <cassert>

Scenario::Scenario(std::string name)
    :m_name (name) {
}

Scenario::Scenario(std::string name, int epoch_count, float keep_rate)
    :m_name(name) {
    m_dropouts.assign(epoch_count, keep_rate);
}

Scenario::Scenario(std::string name,
                   int epoch_count,
                   float keep_begin_rate,
                   float keep_end_rate,
                   std::function<int(int)> generator)
    :m_name(name) {

    bool incremental = keep_end_rate > keep_begin_rate;

    float diff = generator(epoch_count);
    float keep_diff = keep_end_rate - keep_begin_rate;
    if (!incremental) {
        keep_diff *= -1;
    }
    float scale = diff / keep_diff;
    for (int i = 0; i < epoch_count; i++) {
        int generated = generator(i);
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
                   float keep_begin_rate,
                   float keep_end_rate,
                   std::function<int(int)> generator)
    :m_name(name) {
    m_dropouts.assign(epoch_to_skip, 1.0);

    // generate for epoch_count - epoch_to_skip epochs
    // epoch_to_skip'th rate is keep_begin_rate
    // (epoch_count-1)'th rate is keep_end_rate

    bool incremental = keep_end_rate > keep_begin_rate;

    int n_epoch_to_generate = epoch_count - epoch_to_skip -1;
    int diff = generator(n_epoch_to_generate);
    float keep_diff = keep_end_rate - keep_begin_rate;
    if (!incremental) {
        keep_diff *= -1;
    }
    float scale = diff / (keep_diff);

    m_dropouts.push_back(keep_begin_rate);
    for (int i = epoch_to_skip+1; i < epoch_count; i++) {
        if (incremental) {
            m_dropouts.push_back(keep_begin_rate + generator(i-epoch_to_skip)/scale);
        } else {
            m_dropouts.push_back(keep_begin_rate - generator(i-epoch_to_skip)/scale);
        }
    }
}

float Scenario::getKeepRate(int epoch) const {
    assert(epoch < m_dropouts.size());
    return m_dropouts[epoch];
}

float Scenario::averageDropout() const {
    float sum = 0.0f;
    for (float value : m_dropouts) {
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
