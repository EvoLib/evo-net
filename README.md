# EvoNet

[![Code Quality & Tests](https://github.com/EvoLib/evo-net/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoLib/evo-net/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Project Status: Beta](https://img.shields.io/badge/status-beta-blue.svg)](https://github.com/EvoLib/evo-net)

**EvoNet** is a modular and evolvable neural network core designed for integration
with [EvoLib](https://github.com/EvoLib/evo-lib).  
It supports dynamic topologies, recurrent connections, per-neuron activation, and
structural evolution, with a strong emphasis on clarity and explicit behaviour.

---

## Scope

EvoNet is not a state-of-the-art or general-purpose deep learning framework.

It does not aim to compete with libraries such as PyTorch, TensorFlow, or JAX in terms
of performance, scalability, or training algorithms. Backpropagation, GPU acceleration,
and highly optimised tensor operations are outside the scope of this project.

Instead, EvoNet is designed for evolutionary algorithms, structural mutation, and
exploratory research, with a focus on transparent and explicit implementations rather
than performance optimisation or feature completeness.

EvoNet should be understood as a conceptual and experimental model, not as a
production-grade neural-network engine.

---

## Features

- **Explicit, layer-based topology** with support for skip connections, cycles,
  and recurrent paths
- **Typed neuron roles and connection types** (`NeuronRole`, `ConnectionType`)
- **Topology-aware mutation operations**:
  add/remove neurons and connections, mutate weights, change activations
- **Per-neuron activation functions**, configurable and evolvable
- **Explicit 1-step recurrent state model**, without iterative stabilisation passes
- **Runtime topology growth**, e.g. via `add_neuron` and `add_connection`
- **Debug-friendly architecture** with explicit neuron IDs, labels, roles,
  and directional graphs
- **Designed for evolutionary integration**, not gradient-based training
- **Lightweight and extensible**: pure Python, NumPy-based, no hard dependencies

---

> ⚠️ **Project status: Alpha**  
> Interfaces, APIs, and internal structure may change as the project evolves.

---

## Quick Example

```python
from evonet.core import Nnet

net = Nnet()
net.add_layer()  # Input layer
net.add_layer()  # Output layer

net.add_neuron(layer_idx=0, activation="linear", label="in")
net.add_neuron(layer_idx=1, activation="linear", bias=0.5, label="out")

print(net.calc([1.0]))
```
---

## License

MIT License - see [MIT License](https://github.com/EvoLib/evo-net/tree/main/LICENSE).

