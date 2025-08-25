# SPDX-License-Identifier: MIT
"""
Enumerations used in evolvable neural networks.

Includes:
- Neuron roles (input, hidden, output, bias)
- Connection types (standard, recurrent, modulatory, etc.)
"""

from enum import Enum, auto


class NeuronRole(Enum):
    """
    Role of a neuron in the network.

    - INPUT: Receives external input (no bias, no activation)
    - HIDDEN: Internal processing neuron
    - OUTPUT: Final layer neuron providing network output
    - BIAS: Always-on neuron (typically outputs constant 1.0)
    - BIAS: Internal-only constant additive bias.
    """

    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    BIAS = auto()


class ConnectionType(Enum):
    """
    Type of connection between neurons.

    - STANDARD: Regular forward connection
    - INHIBITORY: Reduces or suppresses target activation
    - EXCITATORY: Amplifies target activation
    - MODULATORY: Alters other connections or gates
    - RECURRENT: Connects to earlier time step (feedback)
    """

    STANDARD = auto()
    INHIBITORY = auto()
    EXCITATORY = auto()
    MODULATORY = auto()
    RECURRENT = auto()
