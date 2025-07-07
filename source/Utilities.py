import numpy as np
import warnings

def KLD(x_true, x_predicted):
    """
    Computes the Kullback-Leibler Divergence (KLD) between two distributions.

    Parameters:
        x_true (array-like): True values (P).
        x_predicted (array-like): Predicted values (Q).

    Returns:
        float: KLD(P || Q)
    """
    x_true = np.array(x_true, dtype=np.float64)
    x_predicted = np.array(x_predicted, dtype=np.float64)
    
    # Handle negative values by taking absolute value
    x_true = np.abs(x_true)
    x_predicted = np.abs(x_predicted)
    
    # Add small constant to avoid log(0) and division by 0
    x_true = x_true + 1e-10
    x_predicted = x_predicted + 1e-10
    
    x_true = x_true / np.sum(x_true)  # Normalize true distribution
    x_predicted = x_predicted / np.sum(x_predicted)  # Normalize predicted distribution
    
    # Compute KLD with additional NaN protection
    result = np.sum(x_true * np.log(x_true / x_predicted))
    
    # Return 0 if result is NaN or infinite
    if np.isnan(result) or np.isinf(result):
        return 0.0
    
    return result



def MSE(x_predicted, x_true):
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.
    """
    x_true = np.array(x_true)
    x_predicted = np.array(x_predicted)

    if len(x_true) != len(x_predicted):
        raise ValueError("Input arrays must have the same length.")

    return np.mean((x_true - x_predicted) ** 2)

def JSD(x_predicted, x_true):
    """
    Computes the Jensen-Shannon Divergence (JSD) between true and predicted values.
    """
    x_true = np.array(x_true)
    x_predicted = np.array(x_predicted)

    if len(x_true) != len(x_predicted):
        raise ValueError("Input arrays must have the same length.")

    m = 0.5 * (x_true + x_predicted)
    return 0.5 * (KLD(x_true, m) + KLD(x_predicted, m))