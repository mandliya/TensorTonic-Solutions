import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len).reshape(-1, 1) #[seq_len, 1]
    two_i = np.arange(0, d_model, 2) #[ceil(d_model/2), 0]
    div_term = np.power(base, two_i / d_model)
    angles = position / div_term #(seq_len, d_model/2)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model//2])
    return pe
    