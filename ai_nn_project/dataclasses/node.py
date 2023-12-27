from dataclasses import dataclass

@dataclass
class Node:
    """A node in a neural network."""
    value: float
    derivative: float
    bias: float
    weights: list[float]