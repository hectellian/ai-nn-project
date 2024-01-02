# Libraries Imports
import matplotlib.pyplot as plt

# Module Imports
from ai_nn_project.models.neuronal_network.multilayer_perceptron import MLP

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