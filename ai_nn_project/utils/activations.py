#!/usr/bin/env python3
"""
Activations Module
==================

This module provides an implementation of various activation functions used in artificial neural networks.

Classes:
    ActivationFunction: Protocol for activation functions.
    Sigmoid: Represents the sigmoid activation function.
    ReLU: Represents the ReLU activation function.
    Linear: Represents the linear activation function.
    
Functions:
    None

Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the neuronal classifiers. The activation functions can be used to perform binary classification tasks on suitable datasets.
    
Example:
    from ai_nn_project.utils.activations import Sigmoid
    sigmoid = Sigmoid()
    sigmoid.activate(0)  # 0.5
    sigmoid.derivative(0)  # 0.25
    
Notes:
    - The module is part of the ai_nn_project and follows its coding standards and architectural design.
    
License:
    MIT License
    
Author:
    Anthony Christoforou
    anthony.christoforou@etu.unige.ch
    
    Nathan Soufiane Vanson
    nathan.vanson@etu.unige.ch
    
    Christian William
    christian.william@etu.unige.ch
    
    Mohammed Massi Rashidi
    mohammed.rashidi@etu.unige.ch
    
References:
    - https://en.wikipedia.org/wiki/Activation_function
    - https://en.wikipedia.org/wiki/Sigmoid_function
    - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    - https://en.wikipedia.org/wiki/Linear_function_(calculus)
    
Last Modified:
    25.01.2024
    
See Also:
    - ai_nn_project.models.neuronal_network.multilayer_perceptron
"""
# Libraries
import numpy as np
from typing import Protocol

# Code
class ActivationFunction(Protocol):
    """
    Activation function protocol.
    """
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        """
        Activation function.

        Args:
            z (np.ndarray): Input vector.

        Returns:
            np.ndarray: Output vector.
        """
        ...
        
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of the activation function.

        Args:
            z (np.ndarray): Input vector.

        Returns:
            np.ndarray: Output vector.
        """
        ...
        
class Sigmoid(ActivationFunction):
    """
    Activation function protocol - Sigmoid.
    """
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            z (np.ndarray): Input vector - Sigmoid.

        Returns:
            np.ndarray: Output vector - Sigmoid.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.

        Args:
            z (np.ndarray): Input vector - Sigmoid.

        Returns:
            np.ndarray: Output vector - Sigmoid.
        """
        return Sigmoid.activate(z) * (1 - Sigmoid.activate(z))
    
class ReLU(ActivationFunction):
    """
    Activation function protocol - ReLU.
    """
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.

        Args:
            z (np.ndarray): Input vector - ReLU.

        Returns:
            np.ndarray: Output vector - ReLU.
        """
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of the ReLU activation function.

        Args:
            z (np.ndarray): Input vector - ReLU.

        Returns:
            np.ndarray: Output vector - ReLU.
        """
        return np.where(z > 0, 1, 0)
    
class Linear(ActivationFunction):
    """
    Activation function protocol - Linear.
    """
    @staticmethod
    def activate(z: np.ndarray) -> np.ndarray:
        """
        Linear activation function.

        Args:
            z (np.ndarray): Input vector - Linear.

        Returns:
            np.ndarray: Output vector - Linear.
        """
        return z
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of the linear activation function.

        Args:
            z (np.ndarray): Input vector - Linear.

        Returns:
            np.ndarray: Output vector - Linear.
        """
        return np.ones_like(z)