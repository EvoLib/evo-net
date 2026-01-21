import numpy as np
import pytest

from evonet.core import Nnet
from evonet.enums import ConnectionType, NeuronRole
from evonet.neuron import Neuron


def _mk_linear_ih_h_o() -> tuple[Nnet, Neuron, Neuron, Neuron]:
    """
    Build a minimal 3-layer net (I-H-O), all linear activations, no auto-connections.

    Returns: (net, n_in, n_hid, n_out)
    """
    net = Nnet()
    net.add_layer(3)  # input, hidden, output

    neurons: list[Neuron] = net.add_neuron(
        layer_idx=0, role=NeuronRole.INPUT, activation="linear", connection_init="none"
    )
    n_in = neurons[0]

    neurons = net.add_neuron(
        layer_idx=1, role=NeuronRole.HIDDEN, activation="linear", connection_init="none"
    )
    n_hid = neurons[0]

    neurons = net.add_neuron(
        layer_idx=2, role=NeuronRole.OUTPUT, activation="linear", connection_init="none"
    )
    n_out = neurons[0]

    return net, n_in, n_hid, n_out


def test_recurrent_self_loop_on_hidden_linear() -> None:
    """
    Dynamics:
        h_t = w_ih * u_t + w_hh * h_{t-1}
        y_t = w_ho * h_t
    With initial h_{-1} = 0 (since last_output starts at 0).
    """
    net, n_in, n_hid, n_out = _mk_linear_ih_h_o()

    w_ih = 0.5
    w_hh = 0.8
    w_ho = 2.0

    # I -> H (standard), H -> O (standard), H -> H (recurrent)
    net.add_connection(n_in, n_hid, weight=w_ih, conn_type=ConnectionType.STANDARD)
    net.add_connection(n_hid, n_out, weight=w_ho, conn_type=ConnectionType.STANDARD)
    net.add_connection(n_hid, n_hid, weight=w_hh, conn_type=ConnectionType.RECURRENT)

    inputs = [1.0, 0.0, 0.0, 1.0]

    # Expected linear recurrence
    h_prev = 0.0
    expected_outputs = []
    for u in inputs:
        h_t = w_ih * u + w_hh * h_prev
        y_t = w_ho * h_t
        expected_outputs.append(y_t)
        h_prev = h_t

    # Run network step-by-step
    actual_outputs = []
    for u in inputs:
        y = net.calc([u])[0]  # single output neuron
        actual_outputs.append(y)

    assert np.allclose(actual_outputs, expected_outputs, atol=1e-12)


def test_recurrent_back_edge_output_to_hidden_linear() -> None:
    """
    Dynamics with a back-edge from Output to Hidden:
        h_t = w_ih * u_t + w_oh * y_{t-1}
        y_t = w_ho * h_t
    Note: y_{t-1} uses last_output from the previous forward pass.
    """
    net, n_in, n_hid, n_out = _mk_linear_ih_h_o()

    w_ih = 1.0
    w_ho = 2.0
    w_oh = 0.7  # back-edge weight (O -> H, recurrent)

    # I -> H (standard), H -> O (standard), O -> H (recurrent back-edge)
    net.add_connection(n_in, n_hid, weight=w_ih, conn_type=ConnectionType.STANDARD)
    net.add_connection(n_hid, n_out, weight=w_ho, conn_type=ConnectionType.STANDARD)
    net.add_connection(n_out, n_hid, weight=w_oh, conn_type=ConnectionType.RECURRENT)

    inputs = [1.0, 0.5, -1.0, 0.0]

    y_prev = 0.0
    expected_outputs = []
    for u in inputs:
        h_t = w_ih * u + w_oh * y_prev
        y_t = w_ho * h_t
        expected_outputs.append(y_t)
        y_prev = y_t

    actual_outputs = []
    for u in inputs:
        y = net.calc([u])[0]
        actual_outputs.append(y)

    assert np.allclose(actual_outputs, expected_outputs, atol=1e-12)


def test_recurrent_delay_two_steps_linear_chain() -> None:
    """
    A minimal sequence test for recurrent delay buffers.

    Network:
        input -> hidden -> output
        hidden has a recurrent self-loop with delay=2

    Expected behaviour (linear activations, weights=1):
        t0 input=1 -> hidden=1, output=1
        t1 input=0 -> hidden=0, output=0   (delay=2 not yet filled)
        t2 input=0 -> hidden=1, output=1   (uses hidden output from t0)
    """
    net = Nnet()
    net.add_layer()  # 0 input
    net.add_layer()  # 1 hidden
    net.add_layer()  # 2 output

    n_in = net.add_neuron(
        layer_idx=0, activation="linear", role=NeuronRole.INPUT, connection_init="none"
    )[0]
    n_h = net.add_neuron(
        layer_idx=1, activation="linear", role=NeuronRole.HIDDEN, connection_init="none"
    )[0]
    n_out = net.add_neuron(
        layer_idx=2, activation="linear", role=NeuronRole.OUTPUT, connection_init="none"
    )[0]

    net.add_connection(n_in, n_h, weight=1.0, conn_type=ConnectionType.STANDARD)
    net.add_connection(n_h, n_out, weight=1.0, conn_type=ConnectionType.STANDARD)

    # Recurrent self-loop with delay=2
    net.add_connection(
        n_h, n_h, weight=1.0, conn_type=ConnectionType.RECURRENT, delay=2
    )

    y0 = net.calc([1.0])[0]
    y1 = net.calc([0.0])[0]
    y2 = net.calc([0.0])[0]

    assert y0 == pytest.approx(1.0)
    assert y1 == pytest.approx(0.0)
    assert y2 == pytest.approx(1.0)


def test_reset_full_clears_delay_history() -> None:
    """Full reset must clear delay buffers, otherwise episodes leak memory."""
    net = Nnet()
    net.add_layer()
    net.add_layer()
    net.add_layer()

    n_in = net.add_neuron(
        layer_idx=0, activation="linear", role=NeuronRole.INPUT, connection_init="none"
    )[0]
    n_h = net.add_neuron(
        layer_idx=1, activation="linear", role=NeuronRole.HIDDEN, connection_init="none"
    )[0]
    n_out = net.add_neuron(
        layer_idx=2, activation="linear", role=NeuronRole.OUTPUT, connection_init="none"
    )[0]

    net.add_connection(n_in, n_h, weight=1.0, conn_type=ConnectionType.STANDARD)
    net.add_connection(n_h, n_out, weight=1.0, conn_type=ConnectionType.STANDARD)
    net.add_connection(
        n_h, n_h, weight=1.0, conn_type=ConnectionType.RECURRENT, delay=2
    )

    _ = net.calc([1.0])[0]  # fills history partially
    net.reset(full=True)

    # After full reset, history is gone -> delay contribution must be 0.0
    y = net.calc([0.0])[0]
    assert y == pytest.approx(0.0)


def test_recurrent_delay_zero_normalizes_to_one() -> None:
    net, n_in, n_hid, _ = _mk_linear_ih_h_o()
    c = net.add_connection(
        n_hid, n_hid, weight=1.0, conn_type=ConnectionType.RECURRENT, delay=0
    )
    assert c is not None
    assert c.delay == 1
