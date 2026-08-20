# MetaLearningPlasticityRules



## Running the Simulator

### Prerequisites

- A C++17-compatible compiler (e.g., `clang++` or `g++`)
- [CMake](https://cmake.org/) ≥ 3.10

On macOS, install CMake via Homebrew if you don't have it:
```bash
brew install cmake
```

### Build

```bash
cd ./simulator
```
From the simulator root:
```bash
cmake -S . -B build
cmake --build build
```
This configures the project into a `build/` directory (kept separate from source, and git-ignored) and compiles the executable.

### Run

```bash
./build/sim
```

This runs a fully-connected 3-neuron network with AMPA synapses, STDP plasticity, and independent Poisson external drive to each neuron. Simulation parameters (duration, timestep, neuron/synapse parameters) are currently set directly in `src/main.cpp`.

### Output

Each run produces two CSV files in the working directory:

| File | Contents |
|---|---|
| `spikes.csv` | Spike times per neuron (`neuron_idx, spike_time`) |
| `weights.csv` | Synaptic weight trajectories, sampled at a fixed interval (`time, synapse_0, synapse_1, ...`) |

These can be loaded directly with `pandas` for analysis or plotting (e.g., raster plots from `spikes.csv`, weight evolution curves from `weights.csv`).

### Alternative: build without CMake

For a quick one-off build without configuring CMake:
```bash
g++ -std=c++17 -Iinclude src/*.cpp -o sim
./sim
```
