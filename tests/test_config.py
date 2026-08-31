from pathlib import Path

import pytest
from pydantic import ValidationError

from evonet import (
    BiasConfig,
    ConnectivityConfig,
    DelayConfig,
    EvoNetConfig,
    RecurrentKind,
    WeightsConfig,
)


def test_evonet_config_preserves_network_configuration_semantics() -> None:
    config = EvoNetConfig(
        dim=[2, 3, 1],
        activation=["linear", "tanh", "sigmoid"],
        connectivity=ConnectivityConfig(
            scope="adjacent",
            density=1.0,
            recurrent=[RecurrentKind.DIRECT],
        ),
        weights=WeightsConfig(
            initializer="normal",
            std=0.5,
            bounds=(-2.0, 2.0),
            init_bounds=(-1.0, 1.0),
        ),
        bias=BiasConfig(
            initializer="fixed",
            value=0.1,
            bounds=(-0.5, 0.5),
        ),
        delay=DelayConfig(initializer="fixed", value=2),
    )

    assert config.weights.std == 0.5
    assert config.weights.bounds == (-2.0, 2.0)
    assert config.weights.init_bounds == (-1.0, 1.0)
    assert config.connectivity.recurrent == [RecurrentKind.DIRECT]


def test_evonet_config_rejects_evolutionary_fields() -> None:
    with pytest.raises(ValidationError):
        EvoNetConfig.model_validate(
            {
                "dim": [2, 1],
                "connectivity": {"scope": "adjacent", "density": 1.0},
                "mutation": {"strategy": "constant", "strength": 0.1},
            }
        )


def test_evonet_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "network.yaml"
    yaml_text = (
        "dim: [2, 3, 1]\n"
        "activation: tanh\n"
        "connectivity:\n"
        "  scope: adjacent\n"
        "  density: 1.0\n"
        "  recurrent: none\n"
        "weights:\n"
        "  initializer: zero\n"
        "bias:\n"
        "  initializer: zero\n"
    )
    path.write_text(yaml_text, encoding="utf-8")

    config = EvoNetConfig.from_yaml(path)

    assert config.dim == [2, 3, 1]
    assert config.connectivity.recurrent == []
