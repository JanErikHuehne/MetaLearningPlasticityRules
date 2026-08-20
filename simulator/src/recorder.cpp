#include "recorder.hpp"
#include <fstream>
#include <stdexcept>

Recorder::Recorder(int n_neurons) : spike_times_(n_neurons) {}

void Recorder::record_spike(int neuron_idx, double t) {
    spike_times_.at(neuron_idx).push_back(t); // .at() throws if neuron_idx is out of range
}

int Recorder::spike_count(int neuron_idx) const {
    return static_cast<int>(spike_times_.at(neuron_idx).size());
}

void Recorder::save_to_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file << "neuron_idx,spike_time\n";
    for (size_t i = 0; i < spike_times_.size(); ++i) {
        for (double t : spike_times_[i]) {
            file << i << "," << t << "\n";
        }
    }
}