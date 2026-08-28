# SPDX-License-Identifier: MIT
"""
Serialization utilities for EvoNet networks.

This module provides functions to save and load EvoNet networks
to and from human-readable YAML files (default) or JSON files.
The serialization preserves the full network topology:
- Layers (index, role, label)
- Neurons (id, activation, bias, role, label)
- Connections (src, dst, weight, recurrent)

YAML is recommended for readability and manual editing.
JSON is provided as a secondary option for interoperability.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from evonet.core import Neuron, Nnet
from evonet.enums import ConnectionType, NeuronRole

FORMAT_NAME = "evonet"
FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def to_dict(net: Nnet) -> dict[str, Any]:
    """
    Convert an EvoNet network into a serializable dictionary.

    Args:
        net (Nnet): The network to convert.

    Returns:
        dict[str, Any]: A nested dictionary representation.
    """
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "layers": [
            {
                "index": i,
                "neurons": [
                    {
                        "id": n.id,
                        "activation": n.activation_name,
                        "bias": n.bias,
                        "role": n.role.name,
                        "label": n.label,
                        "dynamics_name": n.dynamics_name,
                        "dynamics_params": n.dynamics_params,
                        "incoming": [
                            {
                                "source": c.source.id,
                                "target": c.target.id,
                                "weight": c.weight,
                                "type": c.type.name,  # store enum as string
                                "delay": c.delay,
                            }
                            for c in n.incoming
                        ],
                    }
                    for n in layer.neurons
                ],
            }
            for i, layer in enumerate(net.layers)
        ],
    }


def _validate_format(data: dict[str, Any]) -> None:
    """
    Validate serialization metadata while accepting legacy files.

    Legacy files without format metadata remain supported.
    """
    format_name = data.get("format")
    format_version = data.get("format_version")

    if format_name is None and format_version is None:
        return

    if format_name is None or format_version is None:
        raise ValueError(
            "Serialized network must define both 'format' and 'format_version'."
        )

    if format_name != FORMAT_NAME:
        raise ValueError(f"Unsupported serialization format: {format_name!r}.")

    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported EvoNet format version: {format_version!r}. "
            f"Supported version: {FORMAT_VERSION}."
        )


def from_dict(data: dict[str, Any]) -> Nnet:
    """Reconstruct a network from a dictionary created by `to_dict`."""
    _validate_format(data)

    net = Nnet()
    neuron_map: dict[str, Neuron] = {}

    # Rebuild layers and neurons
    for layer_info in data["layers"]:
        net.add_layer()
        for n_info in layer_info["neurons"]:
            n = net.add_neuron(
                activation=n_info["activation"],
                bias=n_info["bias"],
                role=NeuronRole[n_info["role"]],
                label=n_info.get("label", ""),
                dynamics_name=n_info.get("dynamics_name", "standard"),
                dynamics_params=n_info.get("dynamics_params", {}),
                connection_init="none",
            )[0]
            n.id = n_info["id"]
            neuron_map[n.id] = n

    # Rebuild connections
    for layer_info in data["layers"]:
        for n_info in layer_info["neurons"]:
            for c_info in n_info["incoming"]:
                src = neuron_map[c_info["source"]]
                dst = neuron_map[c_info["target"]]
                delay = int(c_info.get("delay", 0))
                conn_type = ConnectionType[c_info["type"]]

                net.add_connection(
                    src,
                    dst,
                    weight=c_info["weight"],
                    conn_type=conn_type,
                    delay=delay,
                )

    return net


# ---------------------------------------------------------------------------
# YAML interface
# ---------------------------------------------------------------------------


def save_yaml(net: Nnet, path: str) -> None:
    """
    Save a network to a YAML file (human-readable).

    Args:
        net (Nnet): The network to save.
        path (str): Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            to_dict(net),
            f,
            sort_keys=False,  # preserve order for readability
            default_flow_style=False,  # block style (YAML best practice)
        )


def load_yaml(path: str) -> Nnet:
    """
    Load a network from a YAML file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Nnet: Reconstructed network.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return from_dict(data)


# ---------------------------------------------------------------------------
# JSON interface
# ---------------------------------------------------------------------------


def save_json(net: Nnet, path: str) -> None:
    """
    Save a network to a JSON file.

    Args:
        net (Nnet): The network to save.
        path (str): Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(net), f, indent=2)


def load_json(path: str) -> Nnet:
    """
    Load a network from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Nnet: Reconstructed network.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data)
