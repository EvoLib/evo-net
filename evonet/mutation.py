# SPDX-License-Identifier: MIT
"""
Mutation operations for evolvable neural networks.

Includes:
- Activation mutation
- Weight and bias mutation (Gaussian noise)
- Structural mutations: add/remove neurons and connections
"""

import random
from collections.abc import Collection
from typing import Optional

import numpy as np

from evonet.activation import ACTIVATIONS, random_function_name
from evonet.connection import Connection
from evonet.core import Nnet
from evonet.enums import ConnectionType, NeuronRole, RecurrentKind
from evonet.neuron import Neuron


def mutate_activation(neuron: Neuron, activations: list[str] | None = None) -> None:
    """
    Assign a new random activation function to a single neuron.

    Args:
        neuron (Neuron): The target neuron to mutate.
        activations (list[str] | None): Optional list of allowed activation names.
    """
    neuron.activation_name = random_function_name(activations)
    neuron.activation = ACTIVATIONS[neuron.activation_name]


def mutate_activations(
    net: Nnet, probability: float = 1.0, activations: list[str] | None = None
) -> None:
    """
    Mutate the activation functions of non-input neurons in the network.

    Args:
        net (Nnet): The target network.
        probability (float): Mutation probability per neuron.
        activations (list[str] | None): Optional subset of allowed activation functions.
    """
    for neuron in net.get_all_neurons():
        if neuron.role != NeuronRole.INPUT and np.random.rand() < probability:
            mutate_activation(neuron, activations)


def mutate_weight(conn: Connection, std: float = 0.1) -> None:
    """Apply Gaussian noise to a connection weight."""
    conn.weight += np.random.normal(0, std)


def mutate_weights(net: Nnet, probability: float = 1.0, std: float = 0.1) -> None:
    """
    Apply Gaussian noise to weights of connections in the network.

    Args:
        net (Nnet): The target network.
        probability (float): Probability to mutate each connection.
        std (float): Standard deviation of the noise.
    """

    for conn in net.get_all_connections():
        if np.random.rand() < probability:
            mutate_weight(conn, std)


def mutate_bias(neuron: Neuron, std: float = 0.1) -> None:
    """Apply Gaussian noise to a neuron's bias value."""
    neuron.bias += np.random.normal(0, std)


def mutate_biases(net: Nnet, probability: float = 1.0, std: float = 0.1) -> None:
    """
    Apply Gaussian noise to biases of neurons in the network.

    Args:
        net (Nnet): The target network.
        probability (float): Mutation probability per neuron.
        std (float): Standard deviation of the noise.
    """
    for neuron in net.get_all_neurons():
        if neuron.role != NeuronRole.INPUT and np.random.rand() < probability:
            mutate_bias(neuron, std)


def add_random_connection(
    net: Nnet,
    allowed_recurrent: Optional[Collection[RecurrentKind | str]] = None,
) -> None:
    """
    Add a valid connection between two randomly chosen neurons.

    Rules:
    - disallow connections into INPUT neurons
    - disallow duplicate (source, target) pairs
    - classify connection type by layer order:
      * src_layer < dst_layer  -> STANDARD
      * src_layer >= dst_layer -> RECURRENT

    Recurrent edges are filtered by allowed_recurrent
    (list of 'direct' | 'lateral' | 'indirect').
    """

    # normalize to set[RecurrentKind]
    kinds: set[RecurrentKind] = set()
    if allowed_recurrent:
        for k in allowed_recurrent:
            kinds.add(RecurrentKind(k) if isinstance(k, str) else k)

    all_neurons = net.get_all_neurons()
    if len(all_neurons) < 2:
        return

    # Build a dict of existing connections
    existing: set[tuple[Neuron, Neuron]] = {
        (c.source, c.target) for c in net.get_all_connections()
    }

    # neuron -> layer index
    layer_of: dict[Neuron, int] = {}
    for layer_idx, layer in enumerate(net.layers):
        for neuron in layer.neurons:
            layer_of[neuron] = layer_idx

    def classify(src: Neuron, dst: Neuron) -> tuple[bool, RecurrentKind | None]:
        src_layer_idx, dst_layer_idx = layer_of[src], layer_of[dst]
        if src is dst:
            return True, RecurrentKind.DIRECT
        if src_layer_idx == dst_layer_idx:
            return True, RecurrentKind.LATERAL
        if src_layer_idx > dst_layer_idx:
            return True, RecurrentKind.INDIRECT
        return False, None  # forward

    candidates: list[tuple[Neuron, Neuron]] = []
    for src in all_neurons:
        for dst in all_neurons:
            if dst.role == NeuronRole.INPUT:
                continue
            if (src, dst) in existing:
                continue

            is_recurrent, kind = classify(src, dst)
            if not is_recurrent:
                candidates.append((src, dst))  # forward always allowed
            elif kind in kinds:
                candidates.append((src, dst))  # recurrent allowed by policy

    if not candidates:
        return

    src, dst = random.choice(candidates)
    src_layer_idx, dst_layer_idx = layer_of[src], layer_of[dst]
    conn_type = (
        ConnectionType.STANDARD
        if src_layer_idx < dst_layer_idx
        else ConnectionType.RECURRENT
    )

    net.add_connection(src, dst, conn_type=conn_type)


def remove_random_connection(net: Nnet) -> None:
    """
    Remove a randomly selected connection from the network.

    Does nothing if no connections are present.
    """
    all_connections = net.get_all_connections()
    if not all_connections:
        return

    conn = random.choice(all_connections)
    conn.source.outgoing.remove(conn)
    conn.target.incoming.remove(conn)


def add_random_neuron(net: Nnet) -> None:
    """
    Insert a new hidden neuron into a random layer.

    If the network has only input/output, a hidden layer is inserted.
    """
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


def remove_random_neuron(net: Nnet) -> None:
    """
    Remove a randomly selected hidden neuron from the network.

    All incoming and outgoing connections are also removed.
    """
    hidden_neurons = [n for n in net.get_all_neurons() if n.role == NeuronRole.HIDDEN]
    if not hidden_neurons:
        return

    neuron = random.choice(hidden_neurons)

    # Remove connections
    for conn in list(neuron.incoming):
        conn.source.outgoing.remove(conn)
    for conn in list(neuron.outgoing):
        conn.target.incoming.remove(conn)

    # Remove from layer
    for layer in net.layers:
        if neuron in layer.neurons:
            layer.neurons.remove(neuron)
            break


def split_connection(net: Nnet, activation: str = "tanh", noise: float = 0.1) -> None:
    """
    Insert a neuron in the middle of an existing connection.

    The old connection is removed, and two new ones are created:
    - from source → new neuron
    - from new neuron → target

    Args:
        net (Nnet): The target network.
        activation (str): Activation function for the new neuron.
        noise (float): Optional noise applied to new weights.
    """
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
