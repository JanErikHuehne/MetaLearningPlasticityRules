#pragma once 


struct NeuronParams {
    double gl; 
    double el; 
    double memc;
    double vt;
    double tau_ampa;
    int refractory_bins;

};


struct PlasticityParams {
    double tau_plus, tau_minus;
    double Aplus, Aminus;
    double alpha_pre, alpha_post;
    double lr;
    double gmax;
};

struct SimConfig {
    double dt;
    int n_bins;
};
