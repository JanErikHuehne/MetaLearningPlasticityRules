#pragma once 

#include <vector>
#include <random>
#include "neuron.hpp"
#include "synapse.hpp"
#include "poisson.hpp"
#include "spike_monitor.hpp"
#include "weight_monitor.hpp"

class Network {
    public:
        Network(double dt);

        void add_neuron(const LIFNeuron& n);
        void add_synapse(const PlasticSynapse& s);
        void add_external_drive(int target_idx, double rate_hz, std::mt19937& gen);

        void run(int n_bins, SpikeMonitor& spike_mon, WeightMonitor& weight_mon, int record_interval_bins);

        const std::vector<LIFNeuron>& neurons() const { return neurons_;};
        const std::vector<PlasticSynapse>& synapses() const {return synapses_;};

    private:
        void build_index(); 

        double dt_;

        std::vector<LIFNeuron> neurons_;
        std::vector<PlasticSynapse> synapses_; 
        std::vector<PoissonSpikeGenerator> drives_;
        std::vector<int> drive_targets_;

        std::vector<std::vector<int>> outgoing_;
        std::vector<std::vector<int>> incoming_;

};