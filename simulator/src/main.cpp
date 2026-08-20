#include <random>
#include <iostream>

#include "params.hpp"
#include "neuron.hpp"
#include "synapse.hpp"
#include "network.hpp"
#include "spike_monitor.hpp"
#include "weight_monitor.hpp"



int main(int argc, char* argv[]) {
    // --- parse CLI args: ./sim <tau_plus> <tau_minus> <Aplus> <Aminus> <alpha_pre> <alpha_post> <seed> <output_prefix>
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <tau_plus> <tau_minus> <Aplus> <Aminus> <alpha_pre> <alpha_post> <seed> <output_prefix>\n";
        return 1;
    }


    double tau_plus    = std::stod(argv[1]);
    double tau_minus   = std::stod(argv[2]);
    double Aplus       = std::stod(argv[3]);
    double Aminus      = std::stod(argv[4]);
    double alpha_pre   = std::stod(argv[5]);
    double alpha_post  = std::stod(argv[6]);

    unsigned int seed  = static_cast<unsigned int>(std::stoul(argv[7]));
    std::string output_prefix = argv[8];

    // --- fixed parameters ---
    const double gmax = 2e-9;
    const double lr   = 1e-3;


    std::mt19937 gen(seed);

    double dt = 1e-5;
    double duration = 10000.0;
    int n_bins = static_cast<int>(duration / dt + 0.5);

    NeuronParams neuron_p{
        10e-9, -60e-3, 200e-12, -50e-3, 5e-3,
        static_cast<int>(5e-3 / dt)
    };

    PlasticityParams stdp_p{
        tau_plus, tau_minus,
        Aplus, Aminus,
        alpha_pre, alpha_post,
        lr,
        gmax
    };


    Network net(dt);
    for (int i = 0; i < 3; ++i) net.add_neuron(LIFNeuron(neuron_p));

    std::uniform_real_distribution<double> w0_dist(0.0, gmax);
    int n_neurons = 3;
    for (int i = 0; i < n_neurons; ++i) {
        for (int j = 0; j < n_neurons; ++j) {
            if (i == j) continue;
            net.add_synapse(PlasticSynapse(i, j, w0_dist(gen), stdp_p));
        }
    }

    for (int i = 0; i < 3; ++i) net.add_external_drive(i, 500.0, gen);

    SpikeMonitor spike_mon(3);
    WeightMonitor weight_mon(static_cast<int>(net.synapses().size()));

    net.run(n_bins, spike_mon, weight_mon, 2000);

    weight_mon.save_to_csv(output_prefix + "_weights.csv");
    std::cout << "Done: " << output_prefix << "\n";
    return 0;




}


/*
int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    double dt = 0.1e-4; // 0.01 ms 
    double duration = 1000.0; // 1 s
    int n_bins = static_cast<int>(duration / dt + 0.5);

    // -- Spike Train Recorder ---
    SpikeMonitor spike_mon(3); // 3 neurons
    WeightMonitor weight_mon(6); // 6 synapses (fully connected)

    int record_interval = 100;
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
    net.run(n_bins, spike_mon, weight_mon, record_interval);

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

    spike_mon.save_to_csv("spikes.csv");
    weight_mon.save_to_csv("weights.csv");
    std::cout << "\nSpike times written to spikes.csv\n";
}
*/

