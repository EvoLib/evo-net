# EvoNet

[![Code Quality & Tests](https://github.com/EvoLib/evo-net/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoLib/evo-net/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Project Status: Beta](https://img.shields.io/badge/status-beta-blue.svg)](https://github.com/EvoLib/evo-net)

**EvoNet** is a modular and evolvable neural network core designed for integration
with [EvoLib](https://github.com/EvoLib/evo-lib).  
It supports dynamic topologies, recurrent connections, per-neuron activation, and
structural evolution, with explicit and deterministic behaviour.

---

## Scope

EvoNet is not a state-of-the-art or general-purpose deep learning framework.

It does not aim to compete with libraries such as PyTorch, TensorFlow, or JAX in terms
of performance, scalability, or training algorithms. Backpropagation, GPU acceleration,
and highly optimised tensor operations are outside the scope of this project.

Instead, EvoNet supports evolutionary algorithms, structural mutation, and
exploratory research, with a focus on transparent and explicit implementations rather
than performance optimisation or feature completeness.

EvoNet is a conceptual and experimental model, not a production-grade neural-network engine.

---

## Overview

- Explicit neural network topology with inspectable structure  
- Support for recurrent connections and temporal dynamics
- Dynamic topology growth during evolution
- Per-neuron activations and stateful neuron dynamics
- Supports evolutionary and structural mutation methods
- Deterministic execution model with explicit state  

---

## Technical Highlights

- Explicit **layer / neuron / connection** objects
- Typed neuron roles (`INPUT`, `HIDDEN`, `OUTPUT`)
- Typed connection kinds (`STANDARD`, `RECURRENT`, …)
- Recurrent connections with integer delay buffers
- Per-neuron activation functions and local dynamics
- Deterministic forward execution (`calc()` = one time step)
- Runtime topology growth (add/remove neurons and connections)
- Structural mutation utilities (weights, biases, activations, topology)
- Deterministic weight & bias vectorization (`get_weights`, `set_weights`)
- YAML / JSON serialization of full topology and parameters
- Graphviz-based network visualization
- Pure Python + NumPy, no hard runtime dependencies

---

## Quick Example

```python
from evonet.core import Nnet
from evonet.enums import NeuronRole

net = Nnet()
net.add_layer()  # Input layer
net.add_layer()  # Output layer

net.add_neuron(layer_idx=0, role=NeuronRole.INPUT, activation="linear", connection_init="none", label="in")
net.add_neuron(layer_idx=1, role=NeuronRole.OUTPUT, activation="linear", bias=0.5, label="out")

y = net.calc([1.0])
print(y)
```

---

## Recurrent Connections & Delay

```python
from evonet.enums import ConnectionType

net.add_connection(
    src,
    dst,
    weight=0.8,
    conn_type=ConnectionType.RECURRENT,
    delay=2,   # uses output from t-2
)
```

Semantics:

- delay = 1 → previous time step
- delay > 1 → buffered history
- delay = 0 is normalized to 1 for recurrent edges

A full reset (net.reset(full=True)) clears all delay buffers.

---

## Neuron Dynamics

Neuron behaviour is defined locally at the neuron level:

```python
net.add_neuron(
    layer_idx=1,
    activation="tanh",
    dynamics_name="leaky",
    dynamics_params={"alpha": 0.2},
)
```

Current built-in dynamics:

- standard – stateless activation
- leaky – leaky integrator using last_output

Dynamics are evaluated inside the neuron, not at network level.

---

## Structural Mutation Utilities

The evonet.mutation module provides mutation operators for:

- mutate weights / biases (Gaussian noise)
- mutate activations (global or layer-specific)
- add/remove random connections
- add/remove random hidden neurons
- control recurrent edge kinds (direct, lateral, indirect)
- neutral or near-zero initialization

EvoNet itself does not decide when or why to mutate — that logic belongs in
EvoLib.

---

## Serialization

Full topology and state can be saved and restored:

```python
net.save("network.yaml")
net = Nnet.load("network.yaml")
```

Supported formats:

- YAML (recommended, human-readable)
- JSON (machine-friendly)

Serialization includes:
layers, neurons, activations, biases, dynamics, connections, types, and delays.

---

## Relationship to EvoLib

- EvoNet: execution, structure, state
- EvoLib: evolution strategy, fitness, mutation scheduling, environments

EvoNet is intentionally usable as a standalone library.
Evolutionary control, fitness evaluation, and mutation scheduling are provided by EvoLib.

---

> **Project status: Beta**  
> Interfaces, APIs, and internal structure may change as the project evolves.

---


## License

MIT License - see [MIT License](https://github.com/EvoLib/evo-net/tree/main/LICENSE).

