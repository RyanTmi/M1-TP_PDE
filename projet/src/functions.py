import numpy as np
import scipy.sparse as sps


def mu(
    x: float | np.ndarray,
    y: float | np.ndarray,
    xlen: float,
    ylen: float,
    p: str | None = None,
) -> float | np.ndarray:
    """
    Compute the mu function given x and y, with an optional partial differential.

    Parameters
    ----------
    x : float or np.ndarray
        The x-coordinate(s).
    y : float or np.ndarray
        The y-coordinate(s).
    xlen : float
        The length of the domain in the x direction.
    ylen : float
        The length of the domain in the y direction.
    p : str or None, optional
        The partial derivative to compute ("x" for the first partial derivative according to x).
        Defaults to None.

    Raises
    ------
    ValueError
        If an invalid value for parameter p is provided.

    Returns
    -------
    float or np.ndarray
        The computed mu or its partial derivative.
    """
    if p is None:
        return 2 + np.sin(2 * np.pi * x / xlen) * np.sin(4 * np.pi * y / ylen)
    elif p == "x":
        return 2 * np.pi * np.cos(2 * np.pi * x / xlen) * np.sin(4 * np.pi * y / ylen) / xlen
    elif p == "y":
        return 4 * np.pi * np.sin(2 * np.pi * x / xlen) * np.cos(4 * np.pi * y / ylen) / ylen
    else:
        raise ValueError("Invalid value for parameter p")


def u_ex(
    x: float | np.ndarray,
    y: float | np.ndarray,
    xlen: float,
    ylen: float,
    alpha: float,
    p: str | None = None,
) -> float | np.ndarray:
    """
    Compute the exact solution function u_ex given x and y, with an optional partial differential.

    Parameters
    ----------
    x : float or np.ndarray
        The x-coordinate(s).
    y : float or np.ndarray
        The y-coordinate(s).
    xlen : float
        The length of the domain in the x direction.
    ylen : float
        The length of the domain in the y direction.
    alpha : float
        The exponent in the solution function.
    p : str or None, optional
        The partial derivative to compute ("yy" for the second partial derivative according to y).
        Defaults to None.

    Raises
    ------
    ValueError
        If an invalid value for parameter p is provided.

    Returns
    -------
    float or np.ndarray
        The computed u_ex or its partial derivative.
    """
    if type(x) is float:
        x_inv = 1 / x if x > 0 else 0
        y_inv = 1 / y if y > 0 else 0
    else:
        x_inv = np.zeros_like(x)
        x_inv[x != 0] = 1 / x[x != 0]
        y_inv = np.zeros_like(y)
        y_inv[y != 0] = 1 / y[y != 0]

    xy_alpha = (x * y) ** alpha

    if p is None:
        return xy_alpha * (x - xlen) * (y - ylen)
    elif p == "x":
        return (y - ylen) * (x * (alpha + 1) - xlen * alpha) * xy_alpha * x_inv
    elif p == "y":
        return (x - xlen) * (y * (alpha + 1) - ylen * alpha) * xy_alpha * y_inv
    elif p == "xx":
        return np.where(
            alpha == 1.0,
            2 * y * (y - ylen),
            alpha * (y - ylen) * (xlen * (1 - alpha) + x * (alpha + 1)) * xy_alpha * x_inv**2,
        )
    elif p == "yy":
        return np.where(
            alpha == 1.0,
            2 * x * (x - xlen),
            alpha * (x - xlen) * (ylen * (1 - alpha) + y * (alpha + 1)) * xy_alpha * y_inv**2,
        )
    else:
        raise ValueError("Invalid value for parameter p")


def source(
    x: float | np.ndarray, y: float | np.ndarray, xlen: float, ylen: float, alpha: float
) -> float | np.ndarray:
    """
    Compute the source function f given x, y, xlen, ylen, and alpha.

    Parameters
    ----------
    x : float or np.ndarray
        The x-coordinate(s).
    y : float or np.ndarray
        The y-coordinate(s).
    xlen : float
        The length of the domain in the x direction.
    ylen : float
        The length of the domain in the y direction.
    alpha : float
        The exponent the function.

    Returns
    -------
    float or np.ndarray
        The computed source function f.
    """
    mup = lambda p=None: mu(x, y, xlen, ylen, p)
    up = lambda p=None: u_ex(x, y, xlen, ylen, alpha, p)
    return -mup() * (up("xx") + up("yy")) - up("x") * mup("x") - up("y") * mup("y") + up()


def l2_norm(u: np.ndarray, m: sps.coo_matrix) -> float:
    """
    Compute the L2 norm of u in the finite element space V_h.

    Parameters
    ----------
    u : np.ndarray
        The vector whose norm is to be computed.
    m : sps.coo_matrix
        The mass matrix associated with the domain.

    Returns
    -------
    float
        The L2 norm of the vector u.
    """
    return np.sqrt(u.T @ m @ u)


def h1_norm(u: np.ndarray, m: sps.coo_matrix, s: sps.coo_matrix) -> float:
    """
    Compute the H1 norm of u in the finite element space V_h.

    Parameters
    ----------
    u : np.ndarray
        The vector whose norm is to be computed.
    m : sps.coo_matrix
        The mass matrix associated with the domain.
    s : sps.coo_matrix
        The stiffness matrix associated with the domain.

    Returns
    -------
    float
        The H1 norm of the vector u.
    """
    return np.sqrt((u.T @ m @ u) + (u.T @ s @ u))
