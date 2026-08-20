#pragma once
#include <vector>
#include <string>

class SpikeMonitor {
public:
    explicit SpikeMonitor(int n_neurons);

    void record_spike(int neuron_idx, double t);

    const std::vector<std::vector<double>>& spike_times() const { return spike_times_; }
    int spike_count(int neuron_idx) const;

    void save_to_csv(const std::string& filename) const;

private:
    std::vector<std::vector<double>> spike_times_; // per neuron: list of spike times
};