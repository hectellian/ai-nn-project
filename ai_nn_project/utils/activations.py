# Libraries
import numpy as np
from typing import Protocol

class ActivationFunction(Protocol):
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        ...
        
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        ...
        
class Sigmoid(ActivationFunction):
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        return Sigmoid.activate(z) * (1 - Sigmoid.activate(z))
    
class ReLU(ActivationFunction):
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1, 0)
    
class Linear(ActivationFunction):
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        return z
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)