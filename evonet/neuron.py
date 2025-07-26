# SPDX-License-Identifier: MIT
"""
Neuron definition for evolvable neural network.

Each neuron holds its activation function, input/output connections, bias value, and a
cached output from the last forward pass.
"""

from __future__ import annotations

from typing import Callable
from uuid import uuid4

from evonet.activation import ACTIVATIONS


class Neuron:
    """
    Represents a single neuron in the network.

    Attributes:
        id (str): Unique identifier for tracking and crossover.
        activation_name (str): Name of the activation function.
        bias (float): Bias value added to incoming inputs.
        incoming (list): Incoming connections (to be filled externally).
        outgoing (list): Outgoing connections (to be filled externally).
        output (float): Cached result after activation.
    """

    def __init__(self, activation: str = "tanh", bias: float = 0.0) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation function: '{activation}'")
        self.id: str = str(uuid4())
        self.activation_name: str = activation
        self.activation: Callable[[float], float] = ACTIVATIONS[activation]
        self.bias: float = bias
        self.incoming: list = []
        self.outgoing: list = []
        self.output: float = 0.0

    def reset(self) -> None:
        """Clears output before each forward pass."""
        self.output = 0.0

    def __repr__(self) -> str:
        return (
            f"<Neuron id={self.id[:6]} "
            f"act={self.activation_name} "
            f"bias={self.bias:.2f}>"
        )
