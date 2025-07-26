# SPDX-License-Identifier: MIT
from enum import Enum, auto


class NeuronRole(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    BIAS = auto()


class ConnectionType(Enum):
    STANDARD = auto()
    INHIBITORY = auto()
    EXCITATORY = auto()
    MODULATORY = auto()
