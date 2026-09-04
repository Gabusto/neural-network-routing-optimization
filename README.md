# Neural Network Routing Optimization

## Overview
This repository contains an academic project focused on developing a feedforward neural network to construct a Minimum Spanning Tree (MST) for routing optimization. The model is applied to a computer network fragment of the Prydniprovska Railway (20 nodes, 28 communication channels) to minimize packet transmission delays[cite: 1].

## Key Features
* **Graph Algorithm Baseline:** Utilized Kruskal's algorithm to generate target output vectors (exact MSTs) for training[cite: 1].
* **Neural Network Architecture:** Designed a two-layer feedforward network with a 28-34-28 configuration using MATLAB Deep Learning Toolbox[cite: 1].
* **Training Algorithms:** Evaluated multiple training algorithms, ultimately selecting the Levenberg-Marquardt (trainlm) algorithm for its optimal balance of speed and Mean Squared Error (MSE) minimization[cite: 1].
* **Data Processing:** Formulated input matrices based on channel delays and target matrices for binary edge classification (1 for inclusion in MST, 0 for exclusion)[cite: 1].

## Tech Stack
* **Language:** MATLAB[cite: 1]
* **Libraries:** Deep Learning Toolbox[cite: 1]
* **Concepts:** Feedforward Neural Networks, Graph Theory (MST, Kruskal's algorithm), Data Simulation, Model Evaluation (MSE, Regression)[cite: 1]

## Results
The trained neural network successfully predicts the inclusion of network edges into the Minimum Spanning Tree with high element-wise accuracy, proving the viability of using ML models for rapid, heuristic routing decisions in telecommunication networks.

## Repository Contents
* `network_routing.m` - The main MATLAB script for dataset generation, neural network training, validation, and testing[cite: 1].
* `Project_Report_UKR.pdf` - The full academic paper detailing the mathematical model, network topology, and experimental results[cite: 1].
