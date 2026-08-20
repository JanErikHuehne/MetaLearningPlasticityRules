cmake_minimum_required(VERSION 3.10)

project(SpikingSim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sim
    src/main.cpp
    src/neuron.cpp
    src/synapse.cpp
    src/poisson.cpp
    src/network.cpp
    src/weight_monitor.cpp
    src/spike_monitor.cpp
    )

target_include_directories(sim PRIVATE include)
