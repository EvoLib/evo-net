# SPDX-License-Identifier: MIT
"""
Connection between neurons in an evolvable neural network.

A connection links a source neuron to a target neuron and transmits a weighted signal.
Supports optional connection types for specialized behaviors (e.g. inhibitory,
recurrent).
"""


from collections import deque
from typing import TYPE_CHECKING, Deque, Optional

from evonet.enums import ConnectionType

if TYPE_CHECKING:
    from evonet.neuron import Neuron


class Connection:
    """
    Represents a directed, weighted connection between two neurons.

    Attributes:
        source (Neuron): The source neuron (presynaptic).
        target (Neuron): The target neuron (postsynaptic).
        weight (float): Multiplicative weight of the transmitted signal.
        delay (int): Delay in discrete time steps (used for recurrent edges).
        type (ConnectionType): Type of connection (e.g. standard, recurrent).
    """

    def __init__(
        self,
        source: "Neuron",
        target: "Neuron",
        weight: float = 1.0,
        delay: int = 0,
        conn_type: ConnectionType = ConnectionType.STANDARD,
    ) -> None:

        if delay < 0:
            raise ValueError("delay must be >= 0")

        # Normalize: recurrent implies delay >= 1
        if conn_type is ConnectionType.RECURRENT and delay == 0:
            delay = 1

        self.source = source
        self.target = target
        self.weight = weight
        self.delay = delay
        self.type = conn_type

        self._history: Optional[Deque[float]] = None
        if self.type is ConnectionType.RECURRENT:
            self._history = deque(maxlen=self.delay)

    def set_delay(self, delay: int) -> None:
        if self.type is not ConnectionType.RECURRENT:
            raise ValueError("set_delay is only valid for recurrent connections")
        if delay <= 0:
            delay = 1
        self.delay = int(delay)
        self._history = deque(maxlen=self.delay)

    def push_source_output(self, value: float) -> None:
        """
        Push the current source output into the delay buffer.

        This should be called once per time step after the network computed outputs.
        """
        if self._history is None:
            return
        self._history.append(float(value))

    def delayed_source_output(self) -> float:
        """
        Return the delayed source output.

        Semantics:
            - delay == 0  -> no delay buffer used (caller decides what to do)
            - delay > 0   -> returns output[t-delay] if available else 0.0
        """
        if self.delay == 0:
            return 0.0

        if self._history is None:
            return 0.0

        if len(self._history) < self.delay:
            return 0.0

        return float(self._history[0])

    def reset_buffer(self) -> None:
        """Clear the internal delay history buffer."""
        if self._history is not None:
            self._history.clear()

    def get_signal(self) -> float:
        """
        Return the weighted signal from the source neuron.

        Returns:
            float: source.output × weight
        """

        return self.source.output * self.weight

    def __repr__(self) -> str:
        """
        Return a concise string representation of the connection.

        Example:
            <Conn abc123 -> def456 w=0.85 d=3 type=standard>
        """

        type_str = self.type.name.lower()
        return (
            f"<Conn {self.source.id[:6]} "
            f"-> {self.target.id[:6]} "
            f"d={self.delay} "
            f"w={self.weight:.2f} type={type_str}>"
        )
