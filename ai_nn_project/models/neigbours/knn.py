#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Module
================================

This module provides an implementation of the K-Nearest Neighbors (KNN) algorithm for classification tasks.

Classes:
    KNN: Represents a K-Nearest Neighbors (KNN)  for binary or multiclass classification.

Functions:
    None

Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the module. The KNN class can be used to perform classification tasks on suitable datasets.

Example:
    from ai_nn_project.models.knn import KNN
    knn = KNN(k=3) # Create a KNN with k=3 neighbors

Notes:
    - The module is part of the ai_nn_project and follows its coding standards and architectural design.
    - This KNN implementation supports both binary and multiclass classification.

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
    - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.
    - https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    - https://scikit-learn.org/stable/modules/neighbors.html

Last Modified:
    25.01.2024
"""

# Libraries Imports
import numpy as np
from joblib import Parallel, delayed

# Code
class KNN:
    """
    A simple K-Nearest Neighbors (KNN) model for classification.

    Attributes:
        k (int): The number of neighbors to consider.
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The training labels.
        mode (str): The mode of the KNN. Can be either 'classification' or 'regression'.
    """
    def __init__(self, k: int = 1, mode: str = 'classification') -> None:
        """
        Args:
            k (int, optional): The number of neighbors to consider. Defaults to 1.
            mode (str, optional): The mode of the KNN. Can be either 'classification' or 'regression'. Defaults to 'classification'.
        """
        self.k = k
        self.mode = mode
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the training data.
        
        Args:
            X (numpy.ndarray): The training data.
            y (numpy.ndarray): The training labels.
        """
        self.X = X
        self.y = np.ravel(y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given data in parallel.
        
        Args:
            X (numpy.ndarray): The data to predict the labels for.
            
        Returns:
            numpy.ndarray: The predicted labels.
        """
        predictions = Parallel(n_jobs=-1)(delayed(self._predict)(x) for x in X) # Parallelize the predictions: https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
        return np.array(predictions)
    
    def _predict(self, x: np.ndarray) -> int:
        """
        Predicts the label for the given data point.
        
        Args:
            x (numpy.ndarray): The data point to predict the label for.
            
        Returns:
            int: The predicted label.
        """
        distances = self._euclidean_distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y[k_indices]

        if self.mode == 'classification':
            return np.bincount(k_nearest_labels).argmax()
        elif self.mode == 'regression':
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Mode must be 'classification' or 'regression'")
    
    def _euclidean_distance(self, x: np.ndarray) -> float:
        """
        Calculates the euclidean distance between two data points.
        
        Args:
            x1 (numpy.ndarray): The first data point.
            x2 (numpy.ndarray): The second data point.
            
        Returns:
            float: The euclidean distance between the two data points.
        """
        return np.sqrt(np.sum((self.X - x) ** 2, axis=1))
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the KNN.

        Returns:
            str: A string representation of the KNN.
        """
        return f'KNN(k={self.k}, mode={self.mode})'
    
    def __str__(self) -> str:
        """
        Returns a string representation of the KNN.

        Returns:
            str: A string representation of the KNN.
        """
        return f'KNN(k={self.k}, mode={self.mode})'