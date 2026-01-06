# Single-Layer Perceptron (ANN) Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasRBerenger/Single-Layer-Perceptron-Artificial-Neural-Network/blob/main/single_layer_perceptron.ipynb)

A low-level implementation of linear classifiers using **pure Python and NumPy**, demonstrating the mathematical foundations of neural networks without high-level frameworks.

**1. Medical Diagnosis System**
Multiclass One-vs-All classifier designed to identify pathologies (Flu, Dengue, Chickenpox) based on an 8-element symptom vector with 6 training samples.

**2. Cursive Digit Recognition**
Binary pattern recognition system for simplified handwritten digits ('0', '1'), analyzing spatial features within a 3x3 pixel grid using 4 training samples.

---

### Technical Specifications

* **Core Logic:** Manual derivation of the **Perceptron Learning Rule (Delta Rule)**: `w_new = w + Δw` (where Δw = learning_rate * error * input).
* **Architecture:** **One-vs-All (OvA)** topology. Deploys independent neuron instances for each class to resolve multiclass problems using binary linear separators.
* **Optimization (Bipolar Encoding):** Inputs and targets use `[-1, 1]` instead of `[0, 1]`. Unlike standard binary, the value `-1` allows the network to mathematically learn from **feature absence**, preventing weight stagnation (null updates) and accelerating convergence.
