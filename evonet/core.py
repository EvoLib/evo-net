# SPDX-License-Identifier: MIT
"""
Core class for evolvable neural networks.

Manages neurons, connections, and forward computation. Prepares mutation, crossover, and
export interfaces.
"""

from __future__ import annotations

import graphviz
import numpy as np

from evonet.connection import Connection
from evonet.enums import ConnectionType, NeuronRole
from evonet.layer import Layer
from evonet.neuron import Neuron


class Nnet:
    """
    Evolvable neural network with explicit topology.

    Attributes:
        connections (list[Connection]): All directed, weighted edges.
        input_neurons (list[Neuron]): Subset of neurons used as input nodes.
        hidden_neurons (list[Neuron]): Subset of neurons used as hidden nodes.
        output_neurons (list[Neuron]): Subset of neurons used as output nodes.
    """

    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add_layer(self, count: int = 1) -> int:
        """
        Add a layer to the network.

        Parameter:
            count (int): Number of layers to add
        """

        if count <= 0:
            raise ValueError("Number of layers must be greater then zero")

        for _ in range(count):
            self.layers.append(Layer())

        return len(self.layers) - 1

    def add_neuron(
        self,
        layer_idx: int | None = None,
        activation: str = "tanh",
        bias: float = 0.0,
        label: str = "",
        role: NeuronRole = NeuronRole.HIDDEN,
        count: int = 1,
        connect_layer: bool = True,
    ) -> Neuron:

        if layer_idx is None:
            layer_idx = len(self.layers) - 1  # Add neuron to last layer

        if layer_idx < 0:
            raise ValueError(f"Expected positiv layerindex: got {layer_idx}")
        if layer_idx >= len(self.layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")

        for _ in range(count):
            neuron = Neuron(activation=activation, bias=bias)
            neuron.role = role
            neuron.label = label

            self.layers[layer_idx].neurons.append(neuron)

            if connect_layer and layer_idx > 0:
                # Finde letzten nicht-leeren Layer vor diesem
                for prev_idx in range(layer_idx - 1, -1, -1):
                    prev_layer = self.layers[prev_idx]
                    if prev_layer.neurons:
                        for prev_neuron in prev_layer.neurons:
                            self.add_connection(prev_neuron, neuron)
                        break
        return neuron

    def add_connection(
        self,
        source: Neuron,
        target: Neuron,
        weight: float | None = None,
        conn_type: ConnectionType = ConnectionType.STANDARD,
    ) -> None:

        if weight is None:
            weight = np.random.randn() * 0.5

        conn = Connection(source, target, weight=weight, conn_type=conn_type)
        source.outgoing.append(conn)
        target.incoming.append(conn)

    def reset(self) -> None:
        """Resets all neurons."""
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.reset()

    def calc(self, input_values: list[float]) -> list[float]:

        self.reset()

        # Set Inputs
        input_layer = self.layers[0]
        assert len(input_layer.neurons) == len(input_values)
        for idx, neuron in enumerate(input_layer.neurons):
            neuron.input = float(input_values[idx])

        # Recurrent
        for layer in self.layers:
            for neuron in layer.neurons:
                for conn in neuron.incoming:
                    if conn.type.name.lower() == "recurrent":
                        neuron.input += conn.source.last_output * conn.weight
                    else:
                        neuron.input += conn.source.output * conn.weight

        for layer in self.layers:
            for neuron in layer.neurons:
                total = neuron.input + neuron.bias
                neuron.output = neuron.activation(total)
                for conn in neuron.outgoing:
                    conn.target.input += conn.weight * neuron.output

        # Return Output
        return [n.output for n in self.layers[-1].neurons]

    def get_all_neurons(self) -> list[Neuron]:
        return [n for layer in self.layers for n in layer.neurons]

    def get_all_connections(self) -> list[Connection]:
        return [c for n in self.get_all_neurons() for c in n.outgoing]

    def __repr__(self) -> str:
        total_neurons = sum(len(layer.neurons) for layer in self.layers)
        input_neurons = len(self.layers[0].neurons) if self.layers else 0
        output_neurons = len(self.layers[-1].neurons) if len(self.layers) > 1 else 0
        hidden_neurons = total_neurons - input_neurons - output_neurons

        total_connections = len(self.get_all_connections())

        return (
            f"<Nnet | {len(self.layers)} layers, "
            f"{total_neurons} neurons (I:{input_neurons} H:{hidden_neurons} "
            f"O:{output_neurons}), "
            f"{total_connections} connections "
        )

    def print_graph(
        self,
        name: str,
        engine: str = "dot",
        labels_on: bool = True,
        colors_on: bool = True,
        thickness_on: bool = False,
        fillcolors_on: bool = False,
    ) -> None:
        """Draws the neural network using Graphviz with free positioning."""

        if not self.layers:
            print("No layers to visualize.")
            return

        dot = graphviz.Digraph(name=name, format="png", engine=engine)
        dot.graph_attr.update(
            bgcolor="white",
            rankdir="LR",
            overlap="prism",
            sep="15",
            ratio="fill",
            splines="spline",
            size="6.68,5!",
            dpi="300",
        )
        dot.node_attr.update(
            shape="circle", style="filled", fixedsize="shape", width="1.8"
        )
        dot.edge_attr.update(arrowsize="0.8")

        # Add neurons with coordinates (x = layer, y = index)
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                if neuron.role.name == "INPUT":
                    fillcolor = "lightblue" if fillcolors_on else "white"
                elif neuron.role.name == "OUTPUT":
                    fillcolor = "orange" if fillcolors_on else "white"
                else:
                    fillcolor = "lightgreen" if fillcolors_on else "white"

                label = (
                    f"{neuron.label or neuron.role.name}({layer_idx})\n"
                    f"In: {neuron.input:.3f}\n"
                    f"Out: {neuron.output:.3f}\n"
                    f"LastOut: {neuron.last_output:.3f}\n"
                    f"Bias: {neuron.bias:.3f}\n"
                    f"{neuron.activation_name}"
                )

                pos = f"{layer_idx},{-neuron_idx}!"
                dot.node(
                    name=neuron.id,
                    label=label,
                    fillcolor=fillcolor,
                    pos=pos,
                )

        # Add edges
        for conn in self.get_all_connections():
            label = f"{conn.weight:.2f}" if labels_on else ""
            color = (
                "green"
                if colors_on and conn.weight >= 0
                else "red" if colors_on else "black"
            )
            penwidth = (
                str(max(1, min(5, abs(conn.weight * 5)))) if thickness_on else "1"
            )
            style = "dashed" if conn.type.name == "RECURRENT" else "solid"

            dot.edge(
                conn.source.id,
                conn.target.id,
                label=label,
                color=color,
                penwidth=penwidth,
                style=style,
            )

        dot.render(name, cleanup=True)
