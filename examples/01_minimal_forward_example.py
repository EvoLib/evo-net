from evonet.core import Nnet
from evonet.enums import NeuronRole


def main() -> None:
    net = Nnet()
    net.add_layer()  # Layer 0: Input
    net.add_layer()  # Layer 1: Output

    net.add_neuron(layer_idx=0, role=NeuronRole.INPUT, activation="linear", lable="in")
    net.add_neuron(
        layer_idx=1, role=NeuronRole.OUTPUT, activation="linear", bias=0.5, lable="out"
    )

    result = net.calc([1.0])
    print(f"Result: {result}")

    print("\nNetwork:", net)
    for n in net.get_all_neurons():
        print(" ", n)
    for c in net.get_all_connections():
        print(" ", c)


if __name__ == "__main__":
    main()
