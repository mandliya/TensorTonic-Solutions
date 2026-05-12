import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # sig(x) = 1 / (1 + e^(-x))
    # sig(x) = e^x / (e^x + 1)
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, (1/(1 + np.exp(-x))), (np.exp(x) / (1 + np.exp(x))))    