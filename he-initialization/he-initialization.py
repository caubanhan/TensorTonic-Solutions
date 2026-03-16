def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    W = np.asarray(W)
    L = np.sqrt(6/fan_in)
    W_ = W * 2 * L - L
    return W_