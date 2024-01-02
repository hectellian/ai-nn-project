#!/usr/bin/env python3
"""
Multilayer Perceptron Module
=================

This module provides an implementation of a Multi-Layer Perceptron (MLP), a class of feedforward artificial neural network. It is designed for binary classification tasks and includes various utility functions for activation and loss calculation.

Classes:
    MLP: Represents a Multi-Layer Perceptron (MLP) neural network for binary classification.

Functions:
    mse_loss: Computes Mean Squared Error Loss.

Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the neuronal classifiers. The MLP class can be used to perform binary classification tasks on suitable datasets.

Example:
    from ai_nn_project.models.classifiers.neuronal_network.multilayer_perceptron import MLP
    mlp = MLP([2, 3, 1])  # Create a MLP with 2 inputs, 3 hidden neurons, and 1 output neuron

Notes:
    - The module is part of the ai_nn_project and follows its coding standards and architectural design.

License:
    MIT License

Author:
    Anthony Christoforou 
    anthony.christoforou@etu.unige.ch

References:
    - F. Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", Cornell Aeronautical Laboratory, 1957.
    - https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/linear_model/_perceptron.py#L9

Last Modified:
    02.01.2024

See Also:
    - ai_nn_project.models.classifiers.linear.perceptron
"""

# Library Imports
import numpy as np
from tqdm.notebook import tqdm

# Module Imports
from ai_nn_project.utils.activations import ActivationFunction
from ai_nn_project.utils.evaluations import mse_loss, precision, recall, f1_score, accuracy


class MLP:
    """
    A simple Multi-Layer Perceptron (MLP) neural network for binary classification.

    Attributes:
        layer_sizes (list): List containing the size of each layer.
        weights (list): List of weight matrices for each layer.
        biases (list): List of bias vectors for each layer.
    """

    def __init__(self, layer_sizes: list, activation_objects: list[ActivationFunction] = None, learning_rate: float = 0.01) -> None:
        """
        Initializes the Multi-Layer Perceptron with random weights and biases.

        Args:
            layer_sizes (list): A list containing the number of neurons in each layer.
            activation_func (callable, optional): The activation function to use. Defaults to sigmoid.
        """
        self.layer_sizes = layer_sizes
        
        if activation_objects is None:
            raise ValueError("No activation functions provided.")
        
        self.activation_objects = activation_objects
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def forward(self, input_data: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Performs the forward propagation through the network.

        Args:
            input_data (numpy.ndarray): The input data for the network.

        Returns:
            A tuple containing the output of the network and the intermediate layer activations.
        """
        activations = [input_data] # List to store the activations for each layer
        for w, b, activation in zip(self.weights, self.biases, self.activation_objects):
            z = np.dot(w, activations[-1]) + b
            activations.append(activation.activate(z))
        return activations[-1], activations

    def compute_loss_derivative(self, output, target):
        """
        Computes the derivative of the loss function with respect to the output.

        Args:
            output (numpy.ndarray): The output from the forward pass.
            target (numpy.ndarray): The target values.

        Returns:
            numpy.ndarray: The derivative of the loss function with respect to the output.
        """
        return output - target

    def backward(self, output: np.ndarray, target: np.ndarray, activations: list) -> None:
        """
        Performs the backward propagation and updates the weights and biases.

        Args:
            output (numpy.ndarray): The output from the forward pass.
            target (numpy.ndarray): The target values.
            activations (list): Activations of all layers from the forward pass.
        """
        loss_derivative = self.compute_loss_derivative(output, target)
        for i in reversed(range(len(self.weights))):
            activation = self.activation_objects[i]
            activation_derivative = activation.derivative(activations[i + 1])
            delta = loss_derivative * activation_derivative
            loss_derivative = np.dot(self.weights[i].T, delta)

            # Ensure that delta is averaged over the batch dimension for bias update
            average_delta = np.mean(delta, axis=1, keepdims=True)
            
            # Gradient descent
            self.weights[i] -= self.learning_rate * np.dot(delta, activations[i].T)
            self.biases[i] -= self.learning_rate * average_delta

    def train(self, training_data: np.ndarray, labels: np.ndarray, epochs: int, batch_size: int, verbose: bool = False) -> list[dict[str, float]]:
        """
        Trains the neural network using the provided training data and labels.

        Args:
            training_data (np.ndarray): The training data.
            labels (np.ndarray): The training labels.
            epochs (int): Number of epochs to train for.
            batch_size (int): The size of each batch for training.
            verbose (bool): If True, prints verbose output. Default is False.
            
        Returns:
            list[dict[str, float]]: A list containing the average metrics for each epoch.
        """
        # Check input layer size matches feature size of training data
        if self.layer_sizes[0] != training_data.shape[1]:
            raise ValueError("Input layer size does not match the number of features in training data.")

        metrics = []
        for epoch in tqdm(range(epochs), desc='Training Progress', leave=False):
            epoch_metrics = {'mse_loss': []}

            # Shuffle the data at the beginning of each epoch
            indices = np.arange(training_data.shape[0])
            np.random.shuffle(indices)
            training_data = training_data[indices]
            labels = labels[indices]

            for start_idx in range(0, len(training_data), batch_size):
                end_idx = min(start_idx + batch_size, len(training_data))
                batch_data = training_data[start_idx:end_idx].T  
                batch_labels = labels[start_idx:end_idx].reshape(1, -1)  # Reshape labels as row vector

                output, activations = self.forward(batch_data)
                self.backward(output, batch_labels, activations)

                # Convert output to binary predictions and calculate metrics
                epoch_metrics['mse_loss'].append(mse_loss(batch_labels, output))

            # Compute average metrics for the epoch
            avg_epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - {avg_epoch_metrics}")
            metrics.append(avg_epoch_metrics)

        return metrics


    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predicts the output for a given input using the trained network.

        Args:
            input_data (numpy.ndarray): Input data for making a prediction.

        Returns:
            numpy.ndarray: The predicted output.
        """
        output, _ = self.forward(input_data)
        return output