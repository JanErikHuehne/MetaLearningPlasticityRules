#pragma once
#include <random>

class PoissonSpikeGenerator {
public:
    PoissonSpikeGenerator(double rate_hz, std::mt19937& gen);
    bool step(double t);

private:
    std::exponential_distribution<double> isi_dist_;
    std::mt19937& gen_;
    double next_spike_time_;
};