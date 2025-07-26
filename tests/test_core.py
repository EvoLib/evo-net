from evonet.connection import Connection
from evonet.core import Nnet
from evonet.neuron import Neuron
from evonet.types import NeuronRole


def test_forward_pass_identity() -> None:
    """Testet ein einfaches Netz mit einem Input und einem Output."""
    net = Nnet()

    # Neuronen erstellen
    n_input = Neuron(activation="linear")
    n_output = Neuron(activation="linear")

    # Netz aufbauen
    net.add_neuron(n_input, role=NeuronRole.INPUT)
    net.add_neuron(n_output, role=NeuronRole.OUTPUT)
    net.add_connection(Connection(n_input, n_output, weight=1.0))

    # Eingabe --> Ausgabe testen
    x = [0.75]
    y = net.calc(x)

    assert isinstance(y, list)
    assert len(y) == 1
    assert abs(y[0] - 0.75) < 1e-6


def test_forward_pass_with_bias() -> None:
    """Testet Netz mit Bias am Output-Neuron."""
    net = Nnet()

    n_input = Neuron(activation="linear")
    n_output = Neuron(activation="linear", bias=0.5)

    net.add_neuron(n_input, role=NeuronRole.INPUT)
    net.add_neuron(n_output, role=NeuronRole.OUTPUT)
    net.add_connection(Connection(n_input, n_output, weight=2.0))

    y = net.calc([1.0])

    assert abs(y[0] - 2.5) < 1e-6  # (1.0 * 2.0) + 0.5 = 2.5
