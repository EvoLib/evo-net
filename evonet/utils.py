# SPDX-License-Identifier: MIT

import numpy as np


def connection_init_value(mode: str) -> float | None:
    """
    Return an initial weight value for a new connection.

    Args:
        mode (str): Initialization mode. One of:
            - "zero": weight = 0.0
            - "near_zero": weight ~ U(-0.05, 0.05)
            - "random": weight ~ N(0, 0.5)
            - "none": no initialization (returns None)

    Returns:
        float | None: Initial connection weight or None.
    """
    if mode == "zero":
        return 0.0
    if mode == "near_zero":
        return np.random.uniform(-0.05, 0.05)
    if mode == "random":
        return np.random.randn() * 0.5
    if mode == "none":
        return None
    raise ValueError(f"Unknown connection_init mode: {mode}")
