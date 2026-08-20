#include "poisson.hpp"

PoissonSpikeGenerator::PoissonSpikeGenerator(double rate_hz, std::mt19937& gen)
    : isi_dist_(rate_hz), gen_(gen), next_spike_time_(0.0)
{
    next_spike_time_ = isi_dist_(gen_);
}

bool PoissonSpikeGenerator::step(double t) {
    if (t >= next_spike_time_) {
        next_spike_time_ += isi_dist_(gen_);
        return true;
    }
    return false;
}