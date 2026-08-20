#include "neuron.hpp"

LIFNeuron::LIFNeuron(const NeuronParams& params) : p(params), v(params.el), g_ampa(0.0) {}

bool LIFNeuron::step(double dt) {

    g_ampa += dt * (- g_ampa / p.tau_ampa);

    if (refractory_counter > 0) {
        --refractory_counter;
        return false;
    }

    v += dt * (-p.gl * (v- p.el) - g_ampa * v) / p.memc;
    if (v > p.vt) {
        v = p.el;
        refractory_counter = p.refractory_bins;
        return true;
    }
    return false;
}

void LIFNeuron::receive_spike(double weight) {
    g_ampa += weight;
}