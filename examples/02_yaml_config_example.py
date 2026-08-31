from pathlib import Path

from evonet import EvoNetConfig, Nnet


def main() -> None:
    config_path = Path(__file__).with_suffix(".yaml")
    config = EvoNetConfig.from_yaml(config_path)
    net = Nnet.from_config(config)

    inputs = [0.25, -0.5]
    outputs = net.calc(inputs)

    print(f"Inputs:  {inputs}")
    print(f"Outputs: {outputs}")
    print(f"Network: {net}")


if __name__ == "__main__":
    main()
