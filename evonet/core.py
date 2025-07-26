# SPDX-License-Identifier: MIT
"""
Core class for evolvable neural networks.

Manages neurons, connections, and forward computation. Prepares mutation, crossover, and
export interfaces.
"""

from __future__ import annotations

import numpy as np

from evonet.connection import Connection
from evonet.neuron import Neuron
from evonet.types import NeuronRole


class Nnet:
    """
    Evolvable neural network with explicit topology.

    Attributes:
        neurons (list[Neuron]): All neurons in the network.
        connections (list[Connection]): All directed, weighted edges.
        input_neurons (list[Neuron]): Subset of neurons used as input nodes.
        output_neurons (list[Neuron]): Subset of neurons used as output nodes.
    """

    def __init__(self) -> None:
        self.neurons: list[Neuron] = []
        self.connections: list[Connection] = []
        self.input_neurons: list[Neuron] = []
        self.output_neurons: list[Neuron] = []

    def add_neuron(self, neuron: Neuron, role: NeuronRole = NeuronRole.HIDDEN) -> None:
        """Adds a neuron to the network and assigns its functional role."""
        self.neurons.append(neuron)
        if role == NeuronRole.INPUT:
            self.input_neurons.append(neuron)
        elif role == NeuronRole.OUTPUT:
            self.output_neurons.append(neuron)

    def add_connection(self, conn: Connection) -> None:
        """Adds a connection and updates neuron references."""
        self.connections.append(conn)
        conn.target.incoming.append(conn)
        conn.source.outgoing.append(conn)

    def reset(self) -> None:
        """Resets output values of all neurons (before forward pass)."""
        for neuron in self.neurons:
            neuron.reset()

    def calc(self, input_values: list[float]) -> list[float]:
        """
        Forward pass through the network.

        Args:
            input_values: values to assign to input neurons

        Returns:
            outputs from output neurons (after activation)
        """
        assert len(input_values) == len(self.input_neurons), "Input size mismatch"
        self.reset()

        # Assign inputs
        for i, value in enumerate(input_values):
            self.input_neurons[i].output = float(value)

        # Topological forward computation (assumes acyclic)
        visited = set(self.input_neurons)
        queue = [n for n in self.neurons if n not in visited]

        while queue:
            progressed = False
            for neuron in queue[:]:
                if all(src.source in visited for src in neuron.incoming):
                    total = sum(c.get_signal() for c in neuron.incoming) + neuron.bias
                    neuron.output = neuron.activation(total)
                    visited.add(neuron)
                    queue.remove(neuron)
                    progressed = True
            if not progressed:
                break  # Prevent infinite loop

        return [n.output for n in self.output_neurons]

    def __repr__(self) -> str:
        return (
            f"<Nnet | {len(self.neurons)} neurons, "
            f"{len(self.connections)} connections>"
        )

    def get_weights(self) -> np.ndarray:
        """Returns all connection weights as a flat NumPy array."""
        import numpy as np

        return np.array([c.weight for c in self.connections], dtype=float)

    def set_weights(self, vector: np.ndarray) -> None:
        """Assigns connection weights from a flat NumPy array."""
        assert len(vector) == len(self.connections), "Weight vector length mismatch"
        for i, c in enumerate(self.connections):
            c.weight = float(vector[i])

    def get_biases(self) -> np.ndarray:
        """Returns all neuron biases as a flat NumPy array."""
        import numpy as np

        return np.array([n.bias for n in self.neurons], dtype=float)

    def set_biases(self, vector: np.ndarray) -> None:
        """Assigns neuron biases from a flat NumPy array."""
        assert len(vector) == len(self.neurons), "Bias vector length mismatch"
        for i, n in enumerate(self.neurons):
            n.bias = float(vector[i])
