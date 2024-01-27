#!/usr/bin/env python3
""" 
Visualisation Module
====================

This module provides an implementation of various visualisation functions used in machine learning.

Classes:
    None
    
Functions:
    visualize_weights: Visualizes the weight matrices of an MLP neural network.

Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the neuronal classifiers. The visualisation functions can be used to visualize the weights of a neural network.
    
Example:
    from ai_nn_project.utils.visualisation import visualize_weights
    visualize_weights(mlp, title='MLP Weights') # Visualize the weights of an MLP
    
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
    - https://matplotlib.org
    
Last Modified:
    25.01.2024
    
See Also:
    - ai_nn_project.models.neuronal_network.multilayer_perceptron
"""

# Libraries Imports
import matplotlib.pyplot as plt

# Module Imports
from ai_nn_project.models.neuronal_network.multilayer_perceptron import MLP

# Code
def visualize_weights(network: MLP, title: str = None) -> None:
    """
    Visualizes the weight matrices of an MLP neural network.
    
    Args:
        network (MLP): The MLP neural network to visualize.
        title (str): Optional title for the visualization.
    """
    num_layers = len(network.weights)
    
    # Create a figure with subplots for each weight matrix
    fig, axes = plt.subplots(1, num_layers, figsize=(40, 10))
    
    # Set the title if provided
    if title:
        fig.suptitle(title)
    
    for i, weight_matrix in enumerate(network.weights):
        ax = axes[i]
        ax.set_title(f'Layer {i} Weights')
        ax.matshow(weight_matrix, cmap='viridis')  # You can change the colormap as desired
        fig.colorbar(ax.matshow(weight_matrix, cmap='viridis'), ax=ax)
    
    plt.show()
    
def plot_accuracy(labels: list, predictions: list, title: str = 'Accuracy') -> None:
    """
    Plots the accuracy metric as it evolves during training.
    
    Args:
        labels (list): List of labels.
        predictions (list): List of predictions.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    plt.figure(figsize=(40, 8))
    plt.plot(labels, label='Labels')
    plt.plot(predictions, label='Predictions')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_metrics(metrics: dict[str, float], title: str = 'Performance Metrics') -> None:
    """
    Plots performance metrics.
    
    Args:
        metrics (dict[str, float]): Dictionary of performance metrics and their corresponding scores.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    plt.figure(figsize=(40, 8))
    plt.bar(metrics.keys(), metrics.values())
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()
    
def plot_metrics_by_epochs(metrics_list: list[dict[str, float]], title: str = 'Performance by Epochs') -> None:
    """
    Plots performance metrics as a function of training epochs.
    
    Args:
        metrics_list (list[dict[str, float]]): List of dictionaries containing performance metrics and their corresponding scores.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    epochs = [i for i in range(len(metrics_list))]
    
    plt.figure(figsize=(40, 8))
    for metric, scores in metrics_list[0].items():
        plt.plot(epochs, [metric['accuracy'] for metric in metrics_list], marker='o', linestyle='-', label=metric)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot performance metrics as a function of training dataset size
def plot_metrics_by_dataset_size(sizes: list, scores: dict[str, float], title: str = 'Performance by Dataset Size') -> None:
    """
    Plots performance metrics as a function of training dataset size.
    
    Args:
        sizes (list): List of training dataset sizes.
        scores (dict[str, float]): Dictionary of performance metrics and their corresponding scores.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    # the score dict is of the form {metric: [score1, score2, ...]}
    plt.figure(figsize=(40, 8))
    for metric, scores in scores.items():
        plt.plot(sizes, scores, marker='o', linestyle='-', label=metric)
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

# Function to plot performance metrics by network complexity (layer sizes)
def plot_metrics_by_complexity(results: list[tuple[dict[str, float | int], float, list[dict[str, float | int]]]], title: str = 'Performance by Network Complexity') -> None:
    """
    Plots performance metrics as a function of network complexity (layer sizes).
    
    Args:
        results (tuple[dict, list, dict]): Tuple containing the results of the complexity experiment.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    complexities = []
    accuracies = []

    for hyperparams, _, metrics_list in results:
        # Calculate the complexity as the sum of neurons in all layers
        complexity = sum(hyperparams['layer_sizes'])
        complexities.append(complexity)
        
        # Extract accuracy from the last entry in the metrics list as an example
        # You might want to adjust this to use a different metric or a different method (e.g., averaging)
        accuracy = metrics_list[-1]['accuracy']  # Assuming 'accuracy' is always present
        accuracies.append(accuracy)

    # Plotting
    plt.figure(figsize=(40, 8))
    plt.scatter(complexities, accuracies, alpha=0.6, color='blue')
    plt.title(title)
    plt.xlabel('Network Complexity (Total Neurons)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
def plot_accuracy_during_training(results: list[tuple[dict[str, float | int], float, list[dict[str, float | int]]]], title: str = 'Accuracy During Training by Pruning Level') -> None:
    """
    Plots the accuracy metric as it evolves during training for different pruning levels.
    
    Args:
        results (list[tuple[dict[str, float | int], float, list[dict[str, float | int]]]]): 
            List containing tuples of hyperparameters (including pruning level), score, and metrics over training epochs.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    plt.figure(figsize=(40, 8))

    # Loop through each result to plot accuracy over epochs for different pruning levels
    for hyperparams, _, metrics_list in results:
        pruning_level = hyperparams.get('pruning_level', 0)  # Default to 0 if not present
        accuracies = [metric['accuracy'] for metric in metrics_list if 'accuracy' in metric]  # Extract accuracies

        # Plot accuracy over epochs/steps
        plt.plot(accuracies, label=f'Pruning Level {pruning_level}%')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
# Function to plot performance metrics by noise level
def plot_metrics_by_noise(noise_levels: list[float], scores: dict[str, float], title='Performance by Noise Level') -> None:
    """
    Plots performance metrics as a function of noise level.
    
    Args:
        noise_levels (list): List of noise levels.
        scores (dict[str, float]): Dictionary of performance metrics and their corresponding scores.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    # the score dict is of the form {metric: [score1, score2, ...]}
    plt.figure(figsize=(40, 8))
    for metric, scores in scores.items():
        print(scores)
        plt.plot(noise_levels, scores, marker='o', linestyle='-', label=metric)
    plt.xlabel('Noise Level')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()