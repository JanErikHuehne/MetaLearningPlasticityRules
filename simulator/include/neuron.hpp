#pragma once
#include "params.hpp"

struct LIFNeuron {
    const NeuronParams& p;

    int refractory_bins; 
    double v, g_ampa;
    int refractory_counter = 0;



explicit LIFNeuron(const NeuronParams& params);
bool step(double dt);
void receive_spike(double weight);
};


