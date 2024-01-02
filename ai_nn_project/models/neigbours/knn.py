#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Module
===============================

This module provides an implementation of the K-Nearest Neighbors (KNN) algorithm for classification tasks.

Classes:
    KNNClassifier: Represents a K-Nearest Neighbors (KNN) classifier for binary or multiclass classification.

Functions:
    None

Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the classifiers module. The KNNClassifier class can be used to perform classification tasks on suitable datasets.

Example:
    from ai_nn_project.models.classifiers.knn import KNNClassifier
    knn = KNNClassifier(k=3)  # Create a KNN classifier with k=3 neighbors

Notes:
    - The module is part of the ai_nn_project and follows its coding standards and architectural design.
    - This KNN implementation supports both binary and multiclass classification.

License:
    MIT License

Author:
    Anthony Christoforou
    anthony.christoforou@etu.unige.ch

References:
    - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.
    - https://scikit-learn.org/stable/modules/neighbors.html

Last Modified:
    02.01.2024

See Also:
    - ai_nn_project.models.classifiers.linear.perceptron
"""

# Libraries Imports
import numpy as np

class KNNClassifier:
    """
    A simple K-Nearest Neighbors (KNN) model for classification.

    Attributes:
        k (int): The number of neighbors to consider.
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The training labels.
    """
    def __init__(self, k: int = 1) -> None:
        """
        Args:
            k (int, optional): The number of neighbors to consider. Defaults to 1.
        """
        self.k = k
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the training data.
        
        Args:
            X (numpy.ndarray): The training data.
            y (numpy.ndarray): The training labels.
        """
        self.X = X
        self.y = y
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given data.
        
        Args:
            X (numpy.ndarray): The data to predict the labels for.
            
        Returns:
            numpy.ndarray: The predicted labels.
        """
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x: np.ndarray) -> int:
        """
        Predicts the label for the given data point.
        
        Args:
            x (numpy.ndarray): The data point to predict the label for.
            
        Returns:
            int: The predicted label.
        """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        return np.bincount(k_nearest_labels).argmax()
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculates the euclidean distance between two data points.
        
        Args:
            x1 (numpy.ndarray): The first data point.
            x2 (numpy.ndarray): The second data point.
            
        Returns:
            float: The euclidean distance between the two data points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def __repr__(self) -> str:
        return f'KNN(k={self.k})'
    
    def __str__(self) -> str:
        return f'KNN(k={self.k})'