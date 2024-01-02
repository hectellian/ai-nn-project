# Libraries Imports
import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Mean Squared Error Loss.
    
    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted values.
        
    Returns:
        numpy.ndarray: The MSE loss.
    """
    return ((y_true - y_pred) ** 2).mean()

def precision(y_true, y_pred):
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


def recall(y_true, y_pred):
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

def f1_score(y_true, y_pred):
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

def accuracy(y_true, y_pred):
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
