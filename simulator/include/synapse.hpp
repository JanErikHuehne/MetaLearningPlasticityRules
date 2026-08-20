#pragma once

#include <algorithm>
#include "params.hpp"
#include "neuron.hpp"

struct PlasticSynapse {

    int pre_idx, post_idx; 
    const PlasticityParams& p; 

    double w;
    double trace_pre = 0.0;
    double trace_post = 0.0;

    double t_last_update = 0.0;

    PlasticSynapse(int pre_idx_, int post_idx_, double w0, const PlasticityParams& params) : pre_idx(pre_idx_), post_idx(post_idx_), p(params), w(w0) {};
    void decay_traces(double t);
    void on_pre(double t, LIFNeuron& post_neuron);
    void on_post(double t);

};
