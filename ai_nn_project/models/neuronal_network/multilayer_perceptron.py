#!/usr/bin/env python3
"""
Multilayer Perceptron Module
============================

This module provides an implementation of a Multi-Layer Perceptron (MLP), a class of feedforward artificial neural network. It is designed for binary classification tasks and includes various utility functions for activation and loss calculation.

Classes:
    MLP: Represents a Multi-Layer Perceptron (MLP) neural network for binary classification.

Functions:
    None

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
    
    Nathan Soufiane Vanson
    nathan.vanson@etu.unige.ch
    
    Christian William
    christian.william@etu.unige.ch
    
    Mohammed Massi Rashidi
    mohammed.rashidi@etu.unige.ch

References:
    - F. Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", Cornell Aeronautical Laboratory, 1957.
    - https://en.wikipedia.org/wiki/Multilayer_perceptron
    - https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/linear_model/_perceptron.py#L9

Last Modified:
    25.01.2024
    
See Also:
    - ai_nn_project.utils.activations
    - ai_nn_project.utils.evaluations
"""

# Library Imports
import numpy as np
from tqdm.notebook import tqdm

# Module Imports
from ai_nn_project.utils.activations import ActivationFunction, Sigmoid, ReLU, Linear
from ai_nn_project.utils.evaluations import mse_loss, cross_entropy_loss, accuracy, precision, recall, f1_score, r2_score, mae_loss, mape_loss

MAX_NORM = 1e-8  # Maximum norm for gradient clipping

# Code
class MLP:
    """
    A simple Multi-Layer Perceptron (MLP) neural network for binary classification.

    Attributes:
        layer_sizes (list): List containing the size of each layer.
        weights (list): List of weight matrices for each layer.
        biases (list): List of bias vectors for each layer.
    """

    def __init__(self, layer_sizes: list, activation_objects: list[ActivationFunction] = None, learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32) -> None:
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
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])] # Initialize weights with He initialization
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        
        self.task_type = self._determine_task_type()
        
    def _determine_task_type(self) -> str:
        """
        Determines the task type based on the output activation function.
        
        Returns:
            str: The task type.
        """
        # Example logic to determine the task type
        if isinstance(self.activation_objects[-1], Sigmoid):
            return "classification"
        else:
            return "regression"

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
        if self.task_type == "regression":
            return output - target
        elif self.task_type == "classification":
            # Assuming binary classification with a sigmoid activation
            epsilon = 1e-15
            output = np.clip(output, epsilon, 1 - epsilon)
            return (output - target) / (output * (1 - output))
        else:
            raise ValueError("Unknown task type")
        
    def evaluate(self, input_data: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Evaluates the model on the provided data and labels.

        Args:
            input_data (numpy.ndarray): The input data.
            labels (numpy.ndarray): The target labels.

        Returns:
            dict[str, float]: A dictionary containing the metrics.
        """
        output, _ = self.forward(input_data.T)
        if self.task_type == "regression":
            output = np.round(output)
            return mse_loss(labels, output.T)
        elif self.task_type == "classification":
            output = np.where(output >= 0.5, 1, 0)
            return cross_entropy_loss(labels, output.T)
        else:
            raise ValueError("Unknown task type")
        

    def backward(self, output: np.ndarray, target: np.ndarray, activations: list) -> None:
        """
        Performs the backward propagation and updates the weights and biases.

        Args:
            output (numpy.ndarray): The output from the forward pass.
            target (numpy.ndarray): The target values.
            activations (list): Activations of all layers from the forward pass.
        """
        error = self.compute_loss_derivative(output, target)
        
        for i in reversed(range(len(self.weights))):
            # Compute the derivative of the activation function
            activation_derivative = self.activation_objects[i].derivative(activations[i + 1])

            # Apply chain rule to calculate the delta for the current layer
            delta = error * activation_derivative

            # Calculate the gradient with respect to weights and biases
            grad_w = np.dot(delta, activations[i].T) / delta.shape[1]  # Average over batch
            grad_b = np.mean(delta, axis=1, keepdims=True)
            
            grad_w_norm = np.linalg.norm(grad_w)
            if grad_w_norm > MAX_NORM:
                grad_w = grad_w * MAX_NORM / grad_w_norm

            # Update weights and biases using gradient descent
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

            # Propagate the error backwards to the previous layer
            if i > 0:
                error = np.dot(self.weights[i].T, delta)

    def fit(self, training_data: np.ndarray, labels: np.ndarray, validation_data: np.ndarray | None = None, validation_labels: np.ndarray | None = None, early_stopping_rounds: int = 5, verbose: bool = False) -> list[dict[str, float]]:
        """
        Trains the neural network using the provided training data and labels.

        Args:
            training_data (np.ndarray): The training data.
            labels (np.ndarray): The training labels.
            validation_data (np.ndarray, optional): The validation data. Defaults to None.
            validation_labels (np.ndarray, optional): The validation labels. Defaults to None.
            early_stopping_rounds (int, optional): The number of rounds to wait for validation loss to improve before stopping. Defaults to 5.
            verbose (bool): If True, prints verbose output. Default is False.
            
        Returns:
            list[dict[str, float]]: A list containing the average metrics for each epoch.
        """
        # Check input layer size matches feature size of training data
        if self.layer_sizes[0] != training_data.shape[1]:
            raise ValueError("Input layer size does not match the number of features in training data.")

        metrics = []
        best_validation_loss = float('inf')
        best_model_state = None
        no_improvement_count = 0
        
        for epoch in tqdm(range(self.epochs), desc='Training Progress', disable=not verbose):
            epoch_metrics = {'accuracy': [], 'mse_loss': [], 'mae_loss': [], 'mape_loss': [], 'r2_score': []} if self.task_type == "regression" else {'cross_entropy_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

            # Shuffle the data at the beginning of each epoch
            indices = np.arange(training_data.shape[0])
            np.random.shuffle(indices)
            training_data = training_data[indices]
            labels = labels[indices]

            for start_idx in range(0, len(training_data), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(training_data))
                batch_data = training_data[start_idx:end_idx].T  
                batch_labels = labels[start_idx:end_idx].reshape(1, -1)  # Reshape labels as row vector

                output, activations = self.forward(batch_data)
                self.backward(output, batch_labels, activations)

            # Calculate metrics
            output, _ = self.forward(training_data.T)
            if self.task_type == "regression":
                output = np.round(output)
                epoch_metrics['accuracy'].append(accuracy(batch_labels, output.T))
                epoch_metrics['mse_loss'].append(mse_loss(batch_labels, output.T))
                epoch_metrics['mae_loss'].append(mae_loss(batch_labels, output.T))
                epoch_metrics['mape_loss'].append(mape_loss(batch_labels, output.T))
                epoch_metrics['r2_score'].append(r2_score(batch_labels, output.T))
            elif self.task_type == "classification":
                output = np.where(output > 0.5, 1, 0)
                epoch_metrics['cross_entropy_loss'].append(cross_entropy_loss(batch_labels, output.T))
                epoch_metrics['accuracy'].append(accuracy(batch_labels, output.T))
                epoch_metrics['precision'].append(precision(batch_labels, output.T))
                epoch_metrics['recall'].append(recall(batch_labels, output.T))
                epoch_metrics['f1_score'].append(f1_score(batch_labels, output.T))
            else:
                raise ValueError("Unknown task type")
            
            # Early stopping        
            if validation_data is not None and validation_labels is not None:
                validation_loss = self.evaluate(validation_data, validation_labels)
                if verbose:
                    print(f"Epoch {epoch + 1}/{self.epochs} - Validation loss: {validation_loss}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model_state = self.get_state()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= early_stopping_rounds:
                        self.set_state(best_model_state)
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Compute average metrics for the epoch
            avg_epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - {avg_epoch_metrics}")
            metrics.append(avg_epoch_metrics)

        return metrics

    def get_state(self):
        # Assuming weights and biases are stored in lists self.weights and self.biases
        return {'weights': self.weights.copy(), 'biases': self.biases.copy()}

    def set_state(self, state):
        self.weights = state['weights']
        self.biases = state['biases']

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
    
    def __str__(self) -> str:
        """
        Returns a string representation of the MLP.

        Returns:
            str: A string representation of the MLP.
        """
        return f"MLP(layer_sizes={self.layer_sizes}, activation_objects={self.activation_objects}, learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size})"
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the MLP.

        Returns:
            str: A string representation of the MLP.
        """
        return f"MLP(layer_sizes={self.layer_sizes}, activation_objects={self.activation_objects}, learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size})"