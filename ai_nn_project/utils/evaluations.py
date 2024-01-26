#!/usr/bin/env python3
"""
Evaluations Module
==================

This module provides an implementation of various evaluation metrics used in machine learning.

Classes:
    None
    
Functions:
    mse_loss: Computes Mean Squared Error Loss.
    rmse_loss: Computes Root Mean Squared Error Loss.
    mae_loss: Computes Mean Absolute Error Loss.
    mape_loss: Computes Mean Absolute Percentage Error Loss.
    r2_score: Computes R2 Score.
    precision: Calculate the precision of the predictions.
    recall: Calculate the recall of the predictions.
    f1_score: Calculate the F1 score of the predictions.
    accuracy: Calculate the accuracy of the predictions.
    
Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the neuronal classifiers. The evaluation functions can be used to evaluate the performance of a model on a dataset.

Example:
    from ai_nn_project.utils.evaluations import mse_loss
    mse_loss(y_true, y_pred)  # 0.5

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
    - https://en.wikipedia.org/wiki/Mean_squared_error
    - https://en.wikipedia.org/wiki/Mean_absolute_error
    - https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    - https://en.wikipedia.org/wiki/Coefficient_of_determination
    - https://en.wikipedia.org/wiki/Precision_and_recall
    - https://en.wikipedia.org/wiki/F-score
    - https://en.wikipedia.org/wiki/Accuracy_and_precision
    
Last Modified:
    25.01.2024
    
See Also:
    - ai_nn_project.models.neuronal_network.multilayer_perceptron
"""

# Libraries Imports
import numpy as np

# Code
# --- Regression ---
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Mean Squared Error Loss.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The MSE loss.
    """
    return ((y_true - y_pred) ** 2).mean()

def rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Root Mean Squared Error Loss.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The RMSE loss.
    """
    return np.sqrt(mse_loss(y_true, y_pred))

def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Mean Absolute Error Loss.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The MAE loss.
    """
    return np.abs(y_true - y_pred).mean()

def mape_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Mean Absolute Percentage Error Loss.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The MAPE loss.
    """
    epsilon = 1e-10  # Small constant
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes R2 Score.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The R2 score.
    """
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    if denominator == 0:
        # Handle the zero variance case
        return 0  # or some other appropriate value or handling
    return 1 - ((y_true - y_pred) ** 2).sum() / denominator

# --- Classifications --- 
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Binaey Cross Entropy Loss.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The cross entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def precision(y_true, y_pred) -> float:
    """
    Calculate the precision of the predictions.

    Precision is defined as the ratio of true positives to the sum of true and false positives. 
    It is a measure of the accuracy of the positive predictions made by the model.

    Args:
        y_true (numpy.ndarray): The true labels (ground truth).
        y_pred (numpy.ndarray): The predicted labels by the model.

    Returns:
        float: The precision of the predictions. If there are no positive predictions (denominator is 0), returns 0.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0


def recall(y_true, y_pred) -> float:
    """
    Calculate the recall of the predictions.

    Recall, also known as sensitivity, is the ratio of true positives to the sum of true positives and false negatives. 
    It measures the ability of the model to find all the relevant cases (all true positives).

    Args:
        y_true (numpy.ndarray): The true labels (ground truth).
        y_pred (numpy.ndarray): The predicted labels by the model.

    Returns:
        float: The recall of the predictions. If there are no actual positives (denominator is 0), returns 0.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred) -> float:
    """
    Calculate the F1 score of the predictions.

    The F1 score is the harmonic mean of precision and recall, providing a balance between them. 
    It is useful when you need to take both false positives and false negatives into account.

    Args:
        y_true (numpy.ndarray): The true labels (ground truth).
        y_pred (numpy.ndarray): The predicted labels by the model.

    Returns:
        float: The F1 score of the predictions. If both precision and recall are 0, returns 0.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def accuracy(y_true, y_pred) -> float:
    """
    Calculate the accuracy of the predictions.

    Accuracy is the ratio of correctly predicted observations to the total observations. 
    It is a measure of how many predictions made by the model are correct.

    Args:
        y_true (numpy.ndarray): The true labels (ground truth).
        y_pred (numpy.ndarray): The predicted labels by the model.

    Returns:
        float: The accuracy of the predictions.
    """
    return np.mean(y_true == y_pred)
