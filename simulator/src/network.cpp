#include "network.hpp"
#include "spike_monitor.hpp"
#include "weight_monitor.hpp"

Network::Network(double dt) : dt_(dt) {}

void Network::add_synapse(const PlasticSynapse& s) {
    synapses_.push_back(s);
}

void Network::add_neuron(const LIFNeuron& n) {
    neurons_.push_back(n);
}

void Network::add_external_drive(int target_idx, double rate_hz, std::mt19937& gen) {
    drives_.emplace_back(rate_hz, gen);
    drive_targets_.push_back(target_idx);
}

void Network::build_index() {
    outgoing_.assign(neurons_.size(), {});
    incoming_.assign(neurons_.size(), {});

    for (int s = 0; s < static_cast<int>(synapses_.size()); ++s) {
        outgoing_[synapses_[s].pre_idx].push_back(s);
        incoming_[synapses_[s].post_idx].push_back(s);
    }
}

void Network::run(int n_bins,SpikeMonitor& spike_mon, WeightMonitor& weight_mon,  int record_interval_bins = 1000) {
    build_index();
    std::vector<bool> spiked(neurons_.size());
    for (int bin=0; bin < n_bins; ++bin) {
        double t = bin * dt_; 

        // external drive 
        for (size_t d = 0; d < drives_.size(); ++d) {
            if(drives_[d].step(t)) {
                neurons_[drive_targets_[d]].receive_spike(1.5e-9);
            }
        }

        // for all neurons
        for (size_t i = 0; i < neurons_.size(); ++i) {
            spiked[i] = neurons_[i].step(dt_);
             if (spiked[i]) spike_mon.record_spike(static_cast<int>(i), t);
        }

        // propagate spikes through synapses

        for (size_t i = 0; i < neurons_.size(); ++i) {
            if (!spiked[i]) continue;
            for (int s : outgoing_[i]) synapses_[s].on_pre(t, neurons_[synapses_[s].post_idx]);
            for (int s: incoming_[i]) synapses_[s].on_post(t);
        }
        if (bin % record_interval_bins == 0) {
            std::vector<double> weights(synapses_.size());
            for (size_t s = 0; s < synapses_.size(); ++s) weights[s] = synapses_[s].w;
            weight_mon.record(t, weights);

        }
    }
}