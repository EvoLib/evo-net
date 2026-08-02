import pytest

from evonet.core import Nnet
from evonet.enums import NeuronRole


def test_forward_pass_identity() -> None:
    """Tests a minimal feedforward network with identity mapping."""
    net = Nnet()
    net.add_layer()  # Input layer
    net.add_layer()  # Output layer

    net.add_neuron(
        layer_idx=0,
        activation="linear",
        role=NeuronRole.INPUT,
        label="in",
        connection_init="none",
    )
    net.add_neuron(
        layer_idx=1,
        activation="linear",
        role=NeuronRole.OUTPUT,
        label="out",
        connection_init="none",
    )

    src = net.layers[0].neurons[0]
    dst = net.layers[1].neurons[0]

    net.add_connection(src, dst, weight=1.0)

    # Gewicht prüfen
    assert abs(src.outgoing[0].weight - 1.0) < 1e-6

    result = net.calc([0.75])

    assert isinstance(result, list)
    assert len(result) == 1
    assert abs(result[0] - 0.75) < 1e-6


def test_forward_pass_with_bias() -> None:
    """Tests a simple net with bias on the output neuron."""
    net = Nnet()
    net.add_layer()
    net.add_layer()

    net.add_neuron(
        layer_idx=0,
        activation="linear",
        role=NeuronRole.INPUT,
        label="in",
        connection_init="none",
    )
    net.add_neuron(
        layer_idx=1,
        activation="linear",
        role=NeuronRole.OUTPUT,
        bias=0.5,
        label="out",
        connection_init="none",
    )

    src = net.layers[0].neurons[0]
    dst = net.layers[1].neurons[0]

    net.add_connection(src, dst, weight=2.0)

    # Gewicht prüfen
    assert abs(src.outgoing[0].weight - 2.0) < 1e-6

    result = net.calc([1.0])

    assert isinstance(result, list)
    assert len(result) == 1
    assert abs(result[0] - 2.5) < 1e-6  # (1.0 * 2.0) + 0.5 = 2.5


def test_calc_rejects_wrong_input_size() -> None:
    net = Nnet()
    net.add_layer()
    net.add_layer()

    net.add_neuron(
        layer_idx=0,
        role=NeuronRole.INPUT,
        activation="linear",
        connection_init="none",
    )
    net.add_neuron(
        layer_idx=1,
        role=NeuronRole.OUTPUT,
        activation="linear",
        connection_init="none",
    )

    with pytest.raises(
        ValueError,
        match=r"Expected 1 input values, got 2",
    ):
        net.calc([1.0, 2.0])


def test_invalid_input_does_not_modify_network_state() -> None:
    net = Nnet()
    net.add_layer()
    net.add_layer()

    input_neuron = net.add_neuron(
        layer_idx=0,
        role=NeuronRole.INPUT,
        activation="linear",
        connection_init="none",
    )[0]
    output_neuron = net.add_neuron(
        layer_idx=1,
        role=NeuronRole.OUTPUT,
        activation="linear",
        connection_init="none",
    )[0]

    net.add_connection(input_neuron, output_neuron, weight=1.0)

    net.calc([0.75])

    output_before = output_neuron.output
    last_output_before = output_neuron.last_output

    with pytest.raises(ValueError):
        net.calc([0.25, 0.5])

    assert output_neuron.output == output_before
    assert output_neuron.last_output == last_output_before
