import numpy as np

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
    
    x_true = x_true / np.sum(x_true)  # Normalize true distribution
    x_predicted = x_predicted / np.sum(x_predicted)  # Normalize predicted distribution

    return np.sum(x_true * np.log(x_true / x_predicted))



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