#include "synapse.hpp"
#include <cmath>
#include <algorithm>

void PlasticSynapse::decay_traces(double t) {
    double elapsed = t- t_last_update;
    trace_pre *= std::exp(-elapsed / p.tau_plus);
    trace_post +=  std::exp(-elapsed / p.tau_minus);
    t_last_update = t;

}

void PlasticSynapse::on_pre(double t, LIFNeuron& post_neuron) {
    decay_traces(t);
    post_neuron.receive_spike(w);
    trace_pre += 1.0;
    w = std::clamp(w + p.lr * (p.alpha_pre - p.Aminus * trace_post), 0.0, p.gmax);

}

void PlasticSynapse::on_post(double t) {
    decay_traces(t);
    trace_post += 1;
    w = std::clamp(w + p.lr * (p.alpha_post + p.Aplus * trace_pre), 0.0, p.gmax);
}