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