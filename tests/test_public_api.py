import evonet
from evonet import (
    BiasConfig,
    Connection,
    ConnectionType,
    ConnectivityConfig,
    DelayConfig,
    EvoNetConfig,
    EvoNetNeuronDynamicsConfig,
    Layer,
    Neuron,
    NeuronRole,
    Nnet,
    RecurrentKind,
    WeightsConfig,
    add_random_connection,
    add_random_neuron,
    mutate_activations,
    mutate_biases,
    mutate_weights,
    remove_random_connection,
    remove_random_neuron,
)


def test_public_api_exports_expected_names() -> None:
    expected = {
        "BiasConfig",
        "Connection",
        "ConnectivityConfig",
        "DelayConfig",
        "EvoNetConfig",
        "EvoNetNeuronDynamicsConfig",
        "ConnectionType",
        "Layer",
        "Neuron",
        "NeuronRole",
        "Nnet",
        "RecurrentKind",
        "WeightsConfig",
        "add_random_connection",
        "add_random_neuron",
        "mutate_activations",
        "mutate_biases",
        "mutate_weights",
        "remove_random_connection",
        "remove_random_neuron",
    }

    assert set(evonet.__all__) == expected


def test_public_api_symbols_are_importable() -> None:
    symbols = [
        BiasConfig,
        Connection,
        ConnectivityConfig,
        DelayConfig,
        EvoNetConfig,
        EvoNetNeuronDynamicsConfig,
        ConnectionType,
        Layer,
        Neuron,
        NeuronRole,
        Nnet,
        RecurrentKind,
        WeightsConfig,
        add_random_connection,
        add_random_neuron,
        mutate_activations,
        mutate_biases,
        mutate_weights,
        remove_random_connection,
        remove_random_neuron,
    ]

    assert all(symbol is not None for symbol in symbols)
