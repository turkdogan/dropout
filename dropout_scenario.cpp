#include "dropout_scenario.h"
#incude <functional>

Scenario::Scenario(std::string name, int epoch_count)
    :m_name (name) {
    m_dropouts.assign(epoch_count, 1.0);
}

Scenario::Scenario(std::string name, int epoch_count, float keep_rate)
    :m_name(name) {
    m_dropouts.assign(epoch_count, keep_rate);
}

Scenario::Scenario(std::string name, int epoch_count,
                    std::function<float>(float) generator)
    :m_name(name) {
    for (int i = 0; i < epoch_count; i++) {
        m_dropouts.push_back(generator(i+1));
    }
}

Scenario::Scenario(std::string name, int epoch_count, int epoch_to_skip,
                   std::function<float>(float) generator)
    :m_name(name) {
    m_dropouts.assign(epoch_to_skip, 1.0);
    for (int i = epoch_to_skip; i < epoch_count; i++) {
        m_dropouts.push_back(generator(i-epoch_to_skip+1));
    }
}

float Scenario::getKeepRate(int epoch) {
    assert(epoch < m_dropouts.size());
    return m_dropouts[epoch];
}

float Scenario::averageDropout() {
    float sum = 0.0f;
    for (float value : m_dropouts) {
        sum += value;
    }
    return sum / m_dropouts.size();
}

bool Scenario::enable() {
    return m_dropouts.size() > 0;
}
