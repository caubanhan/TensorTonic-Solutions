import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    # x = np.asarray(x, dtype = float)
    # size = len(x.shape)
    # H, W = x.shape[-2:]
    # out = np.sum(x, axis = (size - 1, size - 2)) / (H*W)
    # return out

    x_arr = np.array(x, copy=True, dtype=np.float64)
    if x_arr.ndim not in [3, 4]:
        raise ValueError(f"Input must be 3D (C,H,W) or 4D (N,C,H,W), got {x_arr.ndim}D")
    return np.mean(x_arr, axis=(-2, -1))