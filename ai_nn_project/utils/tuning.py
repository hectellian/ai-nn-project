#!/usr/bin/env python3
"""
Tuning Module
=============

This module provides an implementation of grid search for hyperparameter tuning of models.

Classes:
    None
    
Functions:
    worker_knn: Worker function for KNN grid search to be used with parallel processing.
    worker_mlp: Worker function for MLP grid search to be used with parallel processing.
    parallel_grid_search_knn: Performs parallel grid search on the KNN model.
    parallel_grid_search_mlp: Performs parallel grid search on the MLP model.
    grid_search_knn: Performs grid search on the KNN model.
    grid_search_mlp: Performs grid search on the MLP model.
    
Usage:
    This module is intended to be used as part of the ai_nn_project, specifically within the module. The grid search functions can be used to perform hyperparameter tuning on suitable models.
    
Example:
    from ai_nn_project.utils.tuning import grid_search_knn
    grid_search_knn(mse_loss, X_train, y_train, X_val, y_val, {'k': [1, 3, 5, 7, 9]})  # {'k': 3}, 0.5, [(params, score, time), ...]
    
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
    - https://en.wikipedia.org/wiki/Hyperparameter_optimization
    - https://en.wikipedia.org/wiki/Parallel_computing
    
Last Modified:
    25.01.2024
    
See Also:
    - ai_nn_project.models.neigbours.knn
    - ai_nn_project.models.neuronal_network.multilayer_perceptron
    - ai_nn_project.utils.activations
"""

# Libraries
import time
import numpy as np
from itertools import product
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

# Module Imports
from ai_nn_project.models.neigbours.knn import KNN
from ai_nn_project.utils.activations import ReLU, Sigmoid
from ai_nn_project.models.neuronal_network.multilayer_perceptron import MLP

# Code
def worker_knn(params: dict, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scoring_func: callable, fixed_params: dict) -> tuple:
    """
    Worker function for KNN grid search to be used with parallel processing.

    Args:
        params (dict): The parameters for the KNN model.
        X_train, y_train (np.ndarray): Training data and labels.
        X_val, y_val (np.ndarray): Validation data and labels.
        scoring_func (callable): The scoring function to use.
        fixed_params (dict): Additional fixed parameters for the model.

    Returns:
        tuple: A tuple containing the parameters, score, and execution time.
    """
    model = KNN(**params, **fixed_params)
    start_time = time.time()
    model.fit(X_train, y_train)
    output = model.predict(X_val)
    score = scoring_func(y_val, output.reshape(-1, 1))
    end_time = time.time()
    return params, score, end_time - start_time

def worker_mlp(params: dict, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, fixed_params: dict) -> tuple:
    """
    Worker function for MLP grid search to be used with parallel processing.

    Args:
        params (dict): The parameters for the MLP model.
        X_train, y_train (np.ndarray): Training data and labels.
        X_val, y_val (np.ndarray): Validation data and labels.
        scoring_func (callable): The scoring function to use.
        fixed_params (dict): Additional fixed parameters for the model.

    Returns:
        tuple: A tuple containing the parameters, score, and execution time.
    """
    params['activation_objects'] = [ReLU() for _ in range(len(params['layer_sizes']) - 2)] + [fixed_params['final_activation']]
    model = MLP(**params)
    metrics = model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10, verbose=False)
    score = model.evaluate(X_val, y_val)
    return params, score, metrics

def parallel_grid_search_knn(scoring_func: callable, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, param_grid: dict, fixed_params: dict = {}, n_jobs: int = -1) -> tuple:
    """
    Performs parallel grid search on the KNN model.
    
    Args:
        scoring_func (callable): The scoring function to use.
        X_train, y_train (np.ndarray): Training data and labels.
        X_val, y_val (np.ndarray): Validation data and labels.
        param_grid (dict): The parameters to use for the grid search.
        fixed_params (dict): Additional fixed parameters for the model.
        n_jobs (int): Number of jobs to run in parallel. Defaults to -1 (using all processors).

    Returns:
        tuple: Best parameters, best score, and a list of results.
    """
    param_combinations = [dict(zip(param_grid.keys(), params)) for params in product(*param_grid.values())]
    results = Parallel(n_jobs=n_jobs)(delayed(worker_knn)(params, X_train, y_train, X_val, y_val, scoring_func, fixed_params) for params in tqdm(param_combinations, desc=f"KNN Grid Search"))

    best_params = None
    best_score = -1
    for params, score, _ in results:
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, results

def parallel_grid_search_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, param_grid: dict, fixed_params: dict = {}, n_jobs: int = -1) -> tuple:
    """
    Performs parallel grid search on the MLP model.
    
    Args:
        scoring_func (callable): The scoring function to use.
        X_train, y_train (np.ndarray): Training data and labels.
        X_val, y_val (np.ndarray): Validation data and labels.
        param_grid (dict): The parameters to use for the grid search.
        fixed_params (dict): Additional fixed parameters for the model.
        n_jobs (int): Number of jobs to run in parallel. Defaults to -1 (using all processors).

    Returns:
        tuple: Best parameters, best score, and a list of results.
    """
    param_combinations = [dict(zip(param_grid.keys(), params)) for params in product(*param_grid.values())]
    results = Parallel(n_jobs=n_jobs)(delayed(worker_mlp)(params, X_train, y_train, X_val, y_val, fixed_params) for params in tqdm(param_combinations, desc="MLP Grid Search"))

    best_params = None
    best_score = -1
    for params, score, _ in results:
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, results
 
def grid_search_knn(scoring_func: callable, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, param_grid: dict, fixed_params: dict = {}) -> tuple[dict, float, list]:
    """Performs a grid search on the KNN model.
    
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
    results = []
    best_params = None
    best_score = -1  # Assuming higher score is better; modify if needed

    # Iterate over all combinations of parameters in the grid
    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        
        model = KNN(**params_dict, **fixed_params)
            
        print(params_dict)
        print(model)

        # Train and evaluate the model
        start_time = time.time() # Calculate the time it takes to train the model
        model.fit(X_train, y_train)
        end_time = time.time() 
        
        # Get the output of the model
        output = model.predict(X_val)
        score = scoring_func(y_val, output.reshape(-1, 1)) # Calculate the score
        print(score)
        
        results.append((params_dict, score, end_time - start_time))

        # Update best parameters if current score is better
        if score > best_score:
            best_score = score
            best_params = params_dict

    return best_params, best_score, results
    
def grid_search_mlp(scoring_func: callable, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, param_grid: dict, fixed_params: dict = {}) -> tuple[dict, float, list]:
    """Performs a grid search on the MLP model.
    
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
    results = []
    best_params = None
    best_score = -1  # Assuming higher score is better; modify if needed

    # Iterate over all combinations of parameters in the grid
    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        
        # Add fixed parameters to the model
        params_dict['activation_objects'] = [ReLU() for _ in range(len(params_dict['layer_sizes']) - 2)] + [fixed_params['final_activation']]
        model = MLP(**params_dict)
        print(params_dict)
        print(model)

        # Train and evaluate the model
        start_time = time.time() # Calculate the time it takes to train the model
        model.fit(X_train, y_train)
        end_time = time.time() 
        
        # Get the output of the model
        output = model.predict(X_val.T)
        if isinstance(fixed_params['final_activation'], Sigmoid):
            formated_output = np.where(output >= 0.5, 1, 0)
        else:
            formated_output = np.round(output)
            
        score = scoring_func(y_val, formated_output.T) # Calculate the score
        print(score)
        
        results.append((params_dict, score, end_time - start_time))

        # Update best parameters if current score is better
        if score > best_score:
            best_score = score
            best_params = params_dict

    return best_params, best_score, results