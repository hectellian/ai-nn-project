from dataclasses import dataclass

@dataclass
class Node:
    """A node in a neural network."""
    name: str
    value: float
    bias: float
    weights: list[float]