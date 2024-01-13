# Libraries
from tqdm.notebook import tqdm
from itertools import product
import numpy as np

# Module Imports
from ai_nn_project.models.neuronal_network.multilayer_perceptron import MLP
from ai_nn_project.models.neigbours.knn import KNN
from ai_nn_project.utils.activations import ReLU

def mlp_grid_search(model, X_train, y_train, X_test, y_test, params, metrics, verbose=False):
    """Performs a grid search on the given model.
    
    Args:
        model (object): The model to use.
        X_train (numpy.ndarray): The training data.
        y_train (numpy.ndarray): The training labels.
        X_test (numpy.ndarray): The test data.
        y_test (numpy.ndarray): The test labels.
        params (dict): The parameters to use for the grid search.
        metrics (list): The metrics to use for evaluation.
        verbose (bool, optional): Whether to print the results. Defaults to False.
        
    Returns:
        dict: The results of the grid search.
    """
    results = {}
    for param in tqdm(list(product(*params.values())), desc=f'Grid Search for {model.__class__.__name__} - {param}'):
        param = {key: value for key, value in zip(params.keys(), param)}
        fit_params = {}
        model.set_params(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[tuple(param.values())] = {metric.__name__: metric(y_test, y_pred) for metric in metrics}
    if verbose:
        print("Grid Search Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
    return results
