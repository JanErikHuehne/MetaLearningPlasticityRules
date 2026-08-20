#pragma once
#include <vector>
#include <string>

class WeightMonitor {
public:
    explicit WeightMonitor(int n_synapses);

    void record(double t, const std::vector<double>& weights);

    void save_to_csv(const std::string& filename) const;

private:
    std::vector<double> sample_times_;
    std::vector<std::vector<double>> weight_history_; // per synapse: list of weights over time
};
