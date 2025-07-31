# SPDX-License-Identifier: MIT
from evonet.neuron import Neuron


class Layer:
    def __init__(self) -> None:
        self.neurons: list[Neuron] = []

    def add_neuron(self, neuron: Neuron) -> None:
        self.neurons.append(neuron)
