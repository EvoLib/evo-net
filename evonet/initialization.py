# SPDX-License-Identifier: MIT
"""Build EvoNet networks from typed network configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from evonet.activation import random_function_name
from evonet.config import DelayConfig, EvoNetConfig
from evonet.enums import ConnectionType, NeuronRole

if TYPE_CHECKING:
    from evonet.core import Nnet


def _apply_weights_init(net: Nnet, cfg: EvoNetConfig) -> None:
    weights_cfg = cfg.weights
    size = net.num_weights
    weights: NDArray[np.float64]

    if weights_cfg.initializer is None:
        return

    elif weights_cfg.initializer == "zero":
        weights = np.zeros(size, dtype=float)

    elif weights_cfg.initializer == "uniform":
        # uniform uses init_bounds if present, otherwise bounds
        lo, hi = weights_cfg.init_bounds or weights_cfg.bounds
        weights = np.random.uniform(lo, hi, size=size)

    elif weights_cfg.initializer == "normal":
        assert weights_cfg.std is not None
        weights = np.random.normal(loc=0.0, scale=weights_cfg.std, size=size)

        # clip ONLY if init_bounds explicitly provided
        if weights_cfg.init_bounds is not None:
            lo, hi = weights_cfg.init_bounds
            weights = np.clip(weights, lo, hi)

    else:
        raise ValueError(f"Unknown weights initializer: {weights_cfg.initializer}")

    net.set_weights(weights)


def _apply_bias_init(net: Nnet, cfg: EvoNetConfig) -> None:
    bias_cfg = cfg.bias
    size = net.num_biases
    bias: NDArray[np.float64]

    if bias_cfg.initializer is None:
        return

    elif bias_cfg.initializer == "zero":
        bias = np.zeros(size, dtype=float)

    elif bias_cfg.initializer == "fixed":
        assert bias_cfg.value is not None
        bias = np.full(size, float(bias_cfg.value), dtype=float)

    elif bias_cfg.initializer == "uniform":
        lo, hi = bias_cfg.init_bounds or bias_cfg.bounds
        bias = np.random.uniform(lo, hi, size=size)

    elif bias_cfg.initializer == "normal":
        assert bias_cfg.std is not None
        bias = np.random.normal(loc=0.0, scale=bias_cfg.std, size=size)

        # clip only if init_bounds explicitly provided
        if bias_cfg.init_bounds is not None:
            lo, hi = bias_cfg.init_bounds
            bias = np.clip(bias, lo, hi)

    else:
        raise ValueError(f"Unknown bias initializer: {bias_cfg.initializer}")

    net.set_biases(bias)


def _apply_delay_init(net: Nnet, cfg: EvoNetConfig) -> None:
    """Initialize delay on recurrent connections only."""
    if cfg.delay is None:
        return

    delay_cfg: DelayConfig = cfg.delay

    for connection in net.get_all_connections():
        if connection.type is not ConnectionType.RECURRENT:
            continue

        if delay_cfg.initializer == "uniform" and delay_cfg.bounds is not None:
            assert delay_cfg.bounds is not None
            lo, hi = delay_cfg.bounds
            delay = int(np.random.randint(lo, hi + 1))
        else:
            assert delay_cfg.value is not None
            delay = int(delay_cfg.value)

        connection.set_delay(delay)


def _build_architecture(
    net: Nnet,
    cfg: EvoNetConfig,
    connection_init: Literal["random", "zero", "near_zero", "none"] = "zero",
) -> None:
    """
    Build the EvoNet architecture (layers, neurons, activations) from config.

    Args:
        net (Nnet): The network instance to initialize.
        cfg (EvoNetConfig): Config with architecture definition.
    """
    # Activation functions per layer
    if isinstance(cfg.activation, list):
        activations = cfg.activation
    else:
        # Input layer linear, others same activation
        activations = ["linear"] + [cfg.activation] * (len(cfg.dim) - 1)

    for layer_idx, num_neurons in enumerate(cfg.dim):

        net.add_layer()

        if num_neurons == 0:
            continue

        activation_name = activations[layer_idx]
        if activation_name == "random":
            if cfg.activations_allowed is not None:
                activation_name = random_function_name(cfg.activations_allowed)
            else:
                activation_name = random_function_name()

        if layer_idx == 0:
            role = NeuronRole.INPUT
        elif layer_idx == len(cfg.dim) - 1:
            role = NeuronRole.OUTPUT
        else:
            role = NeuronRole.HIDDEN

        # resolve dynamics per layer
        if cfg.neuron_dynamics is None:
            dynamics_name = "standard"
            dynamics_params = {}
        else:
            dynamics_cfg = cfg.neuron_dynamics[layer_idx]
            dynamics_name = dynamics_cfg.name
            dynamics_params = dynamics_cfg.params or {}

        allowed_kinds = set(cfg.connectivity.recurrent)
        net.add_neuron(
            count=num_neurons,
            activation=activation_name,
            role=role,
            connection_init=connection_init,
            bias=0.0,
            recurrent=allowed_kinds if role != NeuronRole.INPUT else None,
            connection_scope=cfg.connectivity.scope,
            connection_density=cfg.connectivity.density,
            dynamics_name=dynamics_name,
            dynamics_params=dynamics_params,
        )


def _initialize_default(net: Nnet, cfg: EvoNetConfig) -> None:
    """
    Build a standard EvoNet architecture and initialize parameters according to the
    explicit configuration blocks.

    - Topology is created via `_build_architecture(...)`.
    - Weights are initialized using `cfg.weights`.
    - Biases are initialized using `cfg.bias`.
    - Delay (if configured) is initialized using `cfg.delay`.
    """
    _build_architecture(net, cfg, connection_init="zero")
    _apply_delay_init(net, cfg)
    _apply_weights_init(net, cfg)
    _apply_bias_init(net, cfg)


def _initialize_unconnected(net: Nnet, cfg: EvoNetConfig) -> None:
    """Initialize an EvoNet without connections."""
    _build_architecture(net, cfg, connection_init="none")
    _apply_bias_init(net, cfg)


def _initialize_identity(net: Nnet, cfg: EvoNetConfig) -> None:
    """
    Build a feedforward EvoNet and initialize parameters with an identity-like
    structure.

    - Feedforward connections are created normally.
    - Self-recurrent connections are added explicitly.
    - Weights and biases are set to fixed values
      to approximate identity behavior.

    This preset intentionally overrides standard parameter initialization.
    """
    self_loop_weight = 0.8
    alpha = 0.01

    _build_architecture(net, cfg)
    _apply_delay_init(net, cfg)

    net.set_weights(np.zeros(net.num_weights))
    net.set_biases(np.zeros(net.num_biases))

    for neuron in net.get_all_neurons():
        # Small random bias to break symmetry
        neuron.bias = np.random.uniform(-alpha, alpha)
        for connection in neuron.outgoing:
            # Damped self-recurrence: acts like memory cell
            if (
                connection.type == ConnectionType.RECURRENT
                and connection.source.id == connection.target.id
            ):
                connection.weight = self_loop_weight

            # Small random feedforward weight to allow weak stimulus flow
            if connection.type == ConnectionType.STANDARD:
                connection.weight = np.random.uniform(-alpha, alpha)


def initialize_nnet(net: Nnet, cfg: EvoNetConfig) -> None:
    """Initialize ``net`` using the topology preset selected by ``cfg.initializer``."""
    match cfg.initializer:
        case "default":
            _initialize_default(net, cfg)
        case "unconnected":
            _initialize_unconnected(net, cfg)
        case "identity":
            _initialize_identity(net, cfg)
        case _:
            # EvoNetConfig validates this already. Keep this guard for direct callers.
            raise ValueError(f"Unknown EvoNet initializer: {cfg.initializer}")
