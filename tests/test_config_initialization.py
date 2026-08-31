from evonet import (
    BiasConfig,
    ConnectionType,
    ConnectivityConfig,
    DelayConfig,
    EvoNetConfig,
    Nnet,
    RecurrentKind,
    WeightsConfig,
)


def test_nnet_from_config_builds_default_network() -> None:
    config = EvoNetConfig(
        dim=[2, 3, 1],
        activation="tanh",
        connectivity=ConnectivityConfig(scope="adjacent", density=1.0),
        weights=WeightsConfig(initializer="zero"),
        bias=BiasConfig(initializer="fixed", value=0.25),
    )

    net = Nnet.from_config(config)

    assert [len(layer.neurons) for layer in net.layers] == [2, 3, 1]
    assert net.num_weights == 9
    assert all(weight == 0.0 for weight in net.get_weights())
    assert all(bias == 0.25 for bias in net.get_biases())


def test_nnet_from_config_builds_unconnected_network() -> None:
    config = EvoNetConfig(
        dim=[2, 3, 1],
        initializer="unconnected",
        connectivity=ConnectivityConfig(scope="adjacent", density=1.0),
        bias=BiasConfig(initializer="zero"),
    )

    net = Nnet.from_config(config)

    assert [len(layer.neurons) for layer in net.layers] == [2, 3, 1]
    assert net.num_weights == 0


def test_nnet_from_config_initializes_recurrent_delay() -> None:
    config = EvoNetConfig(
        dim=[1, 2, 1],
        connectivity=ConnectivityConfig(
            scope="adjacent",
            density=1.0,
            recurrent=[RecurrentKind.DIRECT],
        ),
        delay=DelayConfig(initializer="fixed", value=3),
    )

    net = Nnet.from_config(config)
    recurrent = [
        connection
        for connection in net.get_all_connections()
        if connection.type is ConnectionType.RECURRENT
    ]

    assert recurrent
    assert all(connection.delay == 3 for connection in recurrent)
