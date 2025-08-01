# SPDX-License-Identifier: MIT
"""
Connection between neurons in evolvable neural network.

A connection links a source neuron to a target neuron with a weight. Supports optional
connection types for future use (e.g. inhibitory).
"""

from typing import TYPE_CHECKING

from evonet.enums import ConnectionType

if TYPE_CHECKING:
    from evonet.neuron import Neuron


class Connection:
    """
    Represents a directed, weighted connection between two neurons.

    Attributes:
        source (Neuron): The source neuron (presynaptic).
        target (Neuron): The target neuron (postsynaptic).
        weight (float): Multiplicative weight of the signal.
        delay (int): Optional delay in steps (not yet used).
        type (ConnectionType): Type of the connection (e.g. excitatory).
    """

    def __init__(
        self,
        source: "Neuron",
        target: "Neuron",
        weight: float = 1.0,
        delay: int = 0,
        conn_type: ConnectionType = ConnectionType.STANDARD,
    ) -> None:

        self.source = source
        self.target = target
        self.weight = weight
        self.delay = delay
        self.type: ConnectionType = conn_type

    def get_signal(self) -> float:
        """Computes the weighted signal from the source neuron."""
        return self.source.output * self.weight

    def __repr__(self) -> str:
        type_str = self.type.name.lower()
        return (
            f"<Conn {self.source.id[:6]} "
            f"-> {self.target.id[:6]} "
            f"w={self.weight:.2f} type={type_str}>"
        )
