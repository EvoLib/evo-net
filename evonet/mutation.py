# SPDX-License-Identifier: MIT
"""
Basic mutation operations for evolvable neural networks.

Supports weight, bias, and structural mutations.
"""

import random

import numpy as np

from evonet.activation import ACTIVATIONS, random_function_name
from evonet.connection import Connection
from evonet.core import Nnet
from evonet.enums import ConnectionType, NeuronRole
from evonet.neuron import Neuron


def mutate_activation(neuron: Neuron, activations: list[str] | None = None) -> None:
    """Assign a new random activation to a single neuron."""
    neuron.activation_name = random_function_name(activations)
    neuron.activation = ACTIVATIONS[neuron.activation_name]


def mutate_activations(
    net: Nnet, probability: float = 1.0, activations: list[str] | None = None
) -> None:
    """
    Randomly mutate activations of non-input neurons in a network.

    Args:
        net (Nnet): The network whose neurons will be mutated.
        probability (float): Per-neuron mutation probability.
        activations (list[str] | None): Allowed activation names; if None, all.
    """
    for neuron in net.get_all_neurons():
        if neuron.role != NeuronRole.INPUT and np.random.rand() < probability:
            mutate_activation(neuron, activations)


def mutate_weight(conn: Connection, std: float = 0.1) -> None:
    conn.weight += np.random.normal(0, std)


def mutate_weights(net: Nnet, probability: float = 1.0, std: float = 0.1) -> None:
    """Applies Gaussian noise to all connection weights."""

    for conn in net.get_all_connections():
        if np.random.rand() < probability:
            mutate_weight(conn, std)


def mutate_bias(neuron: Neuron, std: float = 0.1) -> None:
    neuron.bias += np.random.normal(0, std)


def mutate_biases(net: Nnet, probability: float = 1.0, std: float = 0.1) -> None:
    """Applies Gaussian noise to all neuron biases (except input neurons)."""
    for neuron in net.get_all_neurons():
        if neuron.role != NeuronRole.INPUT and np.random.rand() < probability:
            mutate_bias(neuron, std)


def add_random_connection(net: Nnet) -> None:
    """Creates a new connection between two random Neuronen."""

    all_neurons = net.get_all_neurons()
    if len(all_neurons) < 2:
        return

    tries = 10  # retry limit to avoid infinite loops
    for _ in range(tries):
        src = random.choice(all_neurons)
        dst = random.choice(all_neurons)

        # No recorrent connection on INPUT
        if src == dst and src.role == NeuronRole.INPUT:
            continue

        # No recurrent connections to INPUT
        if dst.role == NeuronRole.INPUT:
            continue

        if any(conn.target == dst for conn in src.outgoing):
            continue

        if src == dst:
            conn_type = ConnectionType.RECURRENT
        else:
            conn_type = ConnectionType.STANDARD

        net.add_connection(src, dst, conn_type=conn_type)
        return


def remove_random_connection(net: Nnet) -> None:
    """Entfernt zufällig eine bestehende Verbindung."""
    all_connections = net.get_all_connections()
    if not all_connections:
        return

    conn = random.choice(all_connections)
    conn.source.outgoing.remove(conn)
    conn.target.incoming.remove(conn)


def add_random_neuron(net: Nnet) -> None:
    """Fügt ein neues Hidden-Neuron in ein zufälliges Layer ein und verbindet es mit
    vorhandenen Neuronen."""
    if len(net.layers) < 2:
        return

    if len(net.layers) == 2:
        net.insert_layer(1)

    # Ziel-Layer wählen (nicht Input, nicht Output)
    candidate_layers = net.layers[1:-1]
    if not candidate_layers:
        return

    layer = random.choice(candidate_layers)

    net.add_neuron(
        layer_idx=net.layers.index(layer),
        activation="tanh",
        role=NeuronRole.HIDDEN,
        connect_layer=True,
    )


def split_connection(net: Nnet, activation: str = "tanh", noise: float = 0.1) -> None:
    """Add Neuron between connection."""
    all_connections = net.get_all_connections()
    if not all_connections:
        return

    conn = random.choice(all_connections)
    src, dst = conn.source, conn.target

    insert_idx = None
    for idx, layer in enumerate(net.layers):
        if src in layer.neurons:
            insert_idx = idx + 1
            break

    if insert_idx is None or insert_idx >= len(net.layers):
        return

    new_neuron = net.add_neuron(
        layer_idx=insert_idx,
        role=NeuronRole.HIDDEN,
        activation=activation,
        connect_layer=False,
    )

    # Set new connections
    src.outgoing.remove(conn)
    dst.incoming.remove(conn)

    weight = 1.0 + np.random.normal(0, noise)
    net.add_connection(src, new_neuron, weight=weight)
    net.add_connection(new_neuron, dst, weight=conn.weight)
