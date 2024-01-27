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
import matplotlib.colors as mcolors

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
    
    # Adjust for networks with a single layer
    if num_layers == 1:
        fig, ax = plt.subplots(figsize=(10, 10))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 10, 10))
    
    if title:
        fig.suptitle(title)
    
    for i, weight_matrix in enumerate(network.weights):
        ax = axes if num_layers == 1 else axes[i]
        cax = ax.matshow(weight_matrix, cmap='viridis')
        ax.set_title(f'Layer {i} Weights')
        fig.colorbar(cax, ax=ax)
    
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
    num_metrics = len(metrics_list[0])
    fig, axes = plt.subplots(num_metrics, 1, figsize=(40, 8 * num_metrics))
    fig.suptitle(title)

    for i, metric in enumerate(metrics_list[0].keys()):
        values = [metrics[metric] for metrics in metrics_list]
        ax = axes[i]
        ax.plot(values, label=metric)
        ax.set_title(metric)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metric Score')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(pad=3.0)
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
        accuracy = metrics_list[-1]['accuracy']
        accuracies.append(accuracy)

    # Plotting
    plt.figure(figsize=(40, 8))
    plt.scatter(complexities, accuracies, alpha=0.6, color='blue')
    plt.title(title)
    plt.xlabel('Network Complexity (Total Neurons)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
def plot_accuracy_during_training(results: list[tuple[dict[str, float | int], float, list[dict[str, float | int]]]], title: str = 'Accuracy During Training by Layer Size') -> None:
    """
    Plots the accuracy metric as it evolves during training based on the complexity of the network.
    
    Args:
        results (list[tuple[dict[str, float | int], float, list[dict[str, float | int]]]]): 
            List containing tuples of hyperparameters, score, and metrics over training epochs.
        title (str): Optional title for the visualization.
        
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(40, 8))  # Create a figure and a set of subplots
    cmap = plt.get_cmap('viridis')

    complexities = [sum(hp['layer_sizes']) for hp, _, _ in results]
    norm = mcolors.Normalize(vmin=min(complexities), vmax=max(complexities))

    for hyperparams, _, metrics_list in results:
        complexity = sum(hyperparams['layer_sizes'])
        accuracies = [metrics['accuracy'] for metrics in metrics_list]
        line, = ax.plot(accuracies, color=cmap(norm(complexity)))

        # Determine the epoch where training stopped
        final_epoch = len(accuracies)

        # Draw a vertical line at the stopping epoch
        # Use a different style or color if stopping was due to pruning
        if final_epoch < hyperparams['epochs']:  # Assuming 'epochs' is a key in hyperparams
            # Pruning occurred
            ax.axvline(x=final_epoch, color=line.get_color(), linestyle='--', linewidth=1, label='Pruning at Epoch {}'.format(final_epoch))
        else:
            # Reached max epochs
            ax.axvline(x=final_epoch, color=line.get_color(), linestyle='-', linewidth=1, label='Max Epochs at {}'.format(final_epoch))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=10)
    cbar.set_label('Network Complexity')

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.grid(True)

    # Optional: Create a custom legend to explain the vertical lines
    # This part can be customized or omitted based on your preferences
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', linestyle='--', linewidth=1),
                    Line2D([0], [0], color='gray', linestyle='-', linewidth=1)]
    ax.legend(custom_lines, ['Pruning', 'Reached Max Epochs'])

    plt.show()
    
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
        plt.plot(noise_levels, scores, marker='o', linestyle='-', label=metric)
    plt.xlabel('Noise Level')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()