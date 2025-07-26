# SPDX-License-Identifier: MIT
"""
Mutation operators for evolvable neural networks.

Includes mutations for weights, biases, and (optionally) activation functions. Structure
mutations will be implemented separately.
"""

import random

from evonet.core import Nnet


def mutate_weights(
    net: Nnet,
    std: float = 0.1,
    mutation_rate: float = 1.0,
    weight_min: float = -5.0,
    weight_max: float = 5.0,
) -> None:
    """
    Applies Gaussian noise to connection weights.

    Args:
        net (Nnet): The network to mutate.
        std (float): Standard deviation of noise.
        mutation_rate (float): Probability per connection to mutate.
        weight_min (float): Lower bound for weights.
        weight_max (float): Upper bound for weights.
    """
    for conn in net.connections:
        if random.random() < mutation_rate:
            noise = random.gauss(0.0, std)
            conn.weight += noise
            conn.weight = max(min(conn.weight, weight_max), weight_min)


def mutate_biases(
    net: Nnet,
    std: float = 0.1,
    mutation_rate: float = 1.0,
    bias_min: float = -5.0,
    bias_max: float = 5.0,
) -> None:
    """
    Applies Gaussian noise to neuron biases.

    Args:
        net (Nnet): The network to mutate.
        std (float): Standard deviation of noise.
        mutation_rate (float): Probability per neuron to mutate.
        bias_min (float): Lower bound for biases.
        bias_max (float): Upper bound for biases.
    """
    for neuron in net.neurons:
        if random.random() < mutation_rate:
            noise = random.gauss(0.0, std)
            neuron.bias += noise
            neuron.bias = max(min(neuron.bias, bias_max), bias_min)
