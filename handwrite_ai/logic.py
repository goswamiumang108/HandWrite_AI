"""McCulloch–Pitts threshold neurons (practical 1)."""
from typing import Sequence


def mp_neuron(inputs: Sequence[int], weights: Sequence[int], threshold: int) -> int:
    """Return a binary neuron output, with integer binary inputs."""
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have equal length")
    if not inputs or any(x not in (0, 1) for x in inputs):
        raise ValueError("Inputs must be a nonempty sequence of binary values")
    return int(sum(x * w for x, w in zip(inputs, weights)) >= threshold)


def gate(name: str, a: int, b: int = 0) -> int:
    if name == "AND":
        return mp_neuron((a, b), (1, 1), 2)
    if name == "OR":
        return mp_neuron((a, b), (1, 1), 1)
    if name == "NAND":
        return mp_neuron((a, b), (-1, -1), -1)
    if name == "NOT":
        return mp_neuron((a,), (-1,), 0)
    raise ValueError("Supported gates: AND, OR, NAND, NOT")
