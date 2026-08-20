#include <random>
#include <iostream>

#include "params.hpp"
#include "neuron.hpp"
#include "synapse.hpp"
#include "network.hpp"
#include "recorder.hpp"

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    double dt = 0.1e-4; // 0.01 ms 
    double duration = 100.0; // 1 s
    int n_bins = static_cast<int>(duration / dt + 0.5);

    // -- Spike Train Recorder ---
    Recorder recorder(3); // 3 neurons


    // --- share d parameters ----
    NeuronParams neuron_p{
        10e-9, //gl
        -60e-3, // el 
        200e-12, // memc 
        -50e-3, // vt 
        5e-3, // tau_ampa 
        static_cast<int>(5e-3 /dt)  // refractory bins
    };

    PlasticityParams stdp_p{
        5e-3, 5e-3, // tau_plus, tau_minus 
        -2.0,-2.0, // Aplus, Aminus
        -0.1, 0.05, // alpha_pre, alpha_post
        1e-3, // lr 
        2e-9 // gmax
    };

    // --- Build Network --- 

    Network net(dt);

    // 3 neurons, all sharing neuron_p 

    for (int i =0; i < 3; ++i) {
        net.add_neuron(LIFNeuron(neuron_p));
    }

    // --- fully connected network (excluding self-connections), random initial weights ---
    std::uniform_real_distribution<double> w0_dist(0.0, 1e-9);

    int n_neurons = 3;
    for (int i = 0; i < n_neurons; ++i) {
        for (int j = 0; j < n_neurons; ++j) {
            if (i == j) continue; // no self-synapses
            double w0 = w0_dist(gen);
            net.add_synapse(PlasticSynapse(i, j, w0, stdp_p));
        }
    }

    for (int i = 0; i < 3; ++i) {
        net.add_external_drive(i, 500.0, gen);
    }

    // --- run ---
    net.run(n_bins, recorder);

    // --- report results ---
    std::cout << "Final synaptic weights:\n";
    for (const auto& syn : net.synapses()) {
        std::cout << "  w[" << syn.pre_idx << "->" << syn.post_idx << "] = "
                  << syn.w << "\n";

    }

    std::cout << "\nFinal neuron states:\n";
    const auto& neurons = net.neurons();
    for (size_t i = 0; i < neurons.size(); ++i) {
        std::cout << "  neuron " << i << ": v = " << neurons[i].v
                  << ", g_ampa = " << neurons[i].g_ampa << "\n";
    }

    recorder.save_to_csv("spikes.csv");
    std::cout << "\nSpike times written to spikes.csv\n";
}

