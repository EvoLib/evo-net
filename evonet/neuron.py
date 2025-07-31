# SPDX-License-Identifier: MIT
"""
Neuron definition for evolvable neural network.

Each neuron holds its activation function, input/output connections, bias value, and a
cached output from the last forward pass.
"""

from typing import Callable
from uuid import uuid4

from evonet.activation import ACTIVATIONS
from evonet.connection import Connection
from evonet.enums import NeuronRole


class Neuron:
    """
    Represents a single neuron in the network.

    Attributes:
        id (str): Unique identifier for tracking.
        activation_name (str): Name of the activation function.
        bias (float): Bias value added to incoming inputs.
        incoming (list): Incoming connections (to be filled externally).
        outgoing (list): Outgoing connections (to be filled externally).
        output (float): Cached result after activation.
        lable (str): An optional lable
    """

    def __init__(
        self, activation: str = "tanh", lable: str = "", bias: float = 0.0
    ) -> None:

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation function: '{activation}'")
        self.id: str = str(uuid4())
        self.role: NeuronRole = NeuronRole.HIDDEN
        self.activation_name: str = activation
        self.activation: Callable[[float], float] = ACTIVATIONS[activation]
        self.bias: float = bias
        self.incoming: list[Connection] = []
        self.outgoing: list[Connection] = []
        self.input: float = 0.0
        self.output: float = 0.0
        self.last_output: float = 0.0
        self.lable = lable

    def reset(self) -> None:
        self.last_output = self.output
        self.output = 0.0
        self.input = 0.0

    def __repr__(self) -> str:
        return (
            f"Neuron id={self.id[:6]} "
            f"act={self.activation_name} "
            f"role={self.role} "
            f"bias={self.bias:.2f} "
            f"input={self.input:0.5f} "
            f"output={self.output:0.5f}"
        )
