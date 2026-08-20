#include "weight_monitor.hpp"
#include <fstream>
#include <stdexcept>

WeightMonitor::WeightMonitor(int n_synapses) : weight_history_(n_synapses) {}

void WeightMonitor::record(double t, const std::vector<double>& weights) {
    sample_times_.push_back(t);
    for (size_t s = 0; s < weights.size(); ++s) {
        weight_history_.at(s).push_back(weights[s]);
    }
}

void WeightMonitor::save_to_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Could not open file: " + filename);

    file << "time";
    for (size_t s = 0; s < weight_history_.size(); ++s) {
        file << ",synapse_" << s;
    }
    file << "\n";

    for (size_t row = 0; row < sample_times_.size(); ++row) {
        file << sample_times_[row];
        for (size_t s = 0; s < weight_history_.size(); ++s) {
            file << "," << weight_history_[s][row];
        }
        file << "\n";
    }
}