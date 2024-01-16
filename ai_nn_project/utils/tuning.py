# Libraries
from tqdm.notebook import tqdm
from itertools import product
import numpy as np

# Module Imports
from ai_nn_project.models.neuronal_network.multilayer_perceptron import MLP
from ai_nn_project.models.neigbours.knn import KNN
from ai_nn_project.utils.activations import ReLU
    
def grid_search(model_class: object, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scoring_func: callable) -> tuple[dict, float]:
    """Performs a grid search on the given model.
    
    Args:
        model (object): The model to use.
        param_grid (dict): The parameters to use for the grid search.
        X_train (numpy.ndarray): The training data.
        y_train (numpy.ndarray): The training labels.
        X_val (numpy.ndarray): The validation data.
        y_val (numpy.ndarray): The validation labels.
        scoring_func (callable): The scoring function to use.
        
    Returns:
        dict: The best parameters.
        float: The best score.
    """
    best_score = None
    best_params = None

    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        model = model_class(**params_dict) 
        
        if isinstance(model, MLP):
            model.fit(X_train, y_train, epochs=params_dict.get('epochs', 100), 
                      batch_size=params_dict.get('batch_size', 32))
        else:
            model.fit(X_train, y_train)
        
        predictions = model.predict(X_val)
        score = scoring_func(y_val, predictions)

        if best_score is None or score > best_score:
            best_score = score
            best_params = params_dict

    return best_params, best_score