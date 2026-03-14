def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    W = np.asarray(W)
    L = np.sqrt(6/(fan_in+fan_out))
    W_ =  W*(2*L) - L
    return W_