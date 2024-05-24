import numpy as np
import scipy.sparse as sps

from mesh import interior


def mesure(vtx: np.ndarray, /) -> float:
    """
    Calculate the measure (area) of a triangle given its vertices.

    Parameters
    ----------
    vtx : np.ndarray
        An array containing the coordinates of the triangle's vertices.
        Shape should be (3, 2).

    Returns
    -------
    float
        The area of the triangle.

    Raises
    ------
    AssertionError
        If `vtx` does not contain exactly 3 vertices.
    """
    assert len(vtx) == 3, "vtx must contain 3 vertices"
    return np.cross(vtx[0] - vtx[1], vtx[0] - vtx[2]) / 2.0


def mass_local(vtx: np.ndarray, e: np.ndarray, /) -> np.ndarray:
    """
    Compute the local mass matrix for a given element.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    e : np.ndarray
        An array containing the indices of the vertices of the element.
        Length should be 3.

    Returns
    -------
    np.ndarray
        The local mass matrix for the element.

    Raises
    ------
    AssertionError
        If `e` does not contain exactly 3 indices.
    """
    assert len(e) == 3, "e must contain 3 indices"
    return mesure(vtx[e]) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12


def stiffness_local(vtx: np.ndarray, e: np.ndarray, /) -> np.ndarray:
    """
    Compute the local stiffness matrix for a given element.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    e : np.ndarray
        An array containing the indices of the vertices of the element.
        Length should be 3.

    Returns
    -------
    np.ndarray
        The local stiffness matrix for the element.
    """
    assert len(e) == 3, "e must contain 3 indices"
    edge_x = np.array(vtx[e[[1, 2, 0]]])
    edge_y = np.array(vtx[e[[2, 0, 1]]])

    n = np.cross([0, 0, 1], np.column_stack([edge_x - edge_y, [0, 0, 0]]))[:, :-1]
    ff = n / np.sum((vtx[e] - edge_x) * n, axis=1)[:, np.newaxis]

    return mesure(vtx[e]) * np.dot(ff, ff.T)


def assemble_global(vtx: np.ndarray, elt: np.ndarray, local, /) -> sps.coo_matrix:
    """
    Assemble the global matrix from local element matrices.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.
    local : function
        A function that computes the local matrix for an element.

    Returns
    -------
    sps.coo_matrix
        The assembled global sparse matrix.
    """
    row = np.zeros(elt.shape[1] ** 2 * elt.shape[0])
    col = np.zeros(elt.shape[1] ** 2 * elt.shape[0])
    data = np.zeros(elt.shape[1] ** 2 * elt.shape[0])

    l = 0
    for e in elt:
        m = local(vtx, e)
        for j in range(elt.shape[1]):
            for k in range(elt.shape[1]):
                row[l] = e[k]
                col[l] = e[j]
                data[l] = m[j, k]
                l += 1

    n = vtx.shape[0]
    return sps.coo_matrix((data, (row, col)), shape=(n, n))


def mass(vtx: np.ndarray, elt: np.ndarray, /) -> sps.coo_matrix:
    """
    Assemble the global mass matrix.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.

    Returns
    -------
    sps.coo_matrix
        The assembled global mass matrix.
    """
    return assemble_global(vtx, elt, mass_local)


def stiffness(vtx: np.ndarray, elt: np.ndarray, /) -> sps.coo_matrix:
    """
    Assemble the global stiffness matrix.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.

    Returns
    -------
    sps.coo_matrix
        The assembled global stiffness matrix.
    """
    return assemble_global(vtx, elt, stiffness_local)


def assemble_linear_system_l(vtx: np.ndarray, elt: np.ndarray, mu: np.ndarray, /) -> sps.coo_matrix:
    """
    Assemble the left-hand side matrix of the linear system.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.
    mu : np.ndarray
        An array containing the values of the parameter mu at the vertices.

    Returns
    -------
    sps.coo_matrix
        The assembled left-hand side matrix of the linear system.
    """
    local = lambda vtx, e: mass_local(vtx, e) + mu[e].sum() * stiffness_local(vtx, e) / 3.0
    return assemble_global(vtx, elt, local)


def assemble_linear_system_r(m: sps.coo_matrix, f: np.ndarray) -> np.ndarray:
    """
    Assemble the right-hand side vector of the linear system.

    Parameters
    ----------
    m : sps.coo_matrix
        The mass matrix.
    f : np.ndarray
        The source term values at the vertices.

    Returns
    -------
    np.ndarray
        The assembled right-hand side vector of the linear system.
    """
    return m @ f


def apply_dirichlet_boundary_conditions(
    a: sps.coo_matrix, f: np.ndarray, vtx: np.ndarray, xlen: float, ylen: float, /
) -> tuple[sps.coo_matrix, np.ndarray]:
    """
    Apply Dirichlet boundary conditions to the linear system.

    Parameters
    ----------
    a : sps.coo_matrix
        The left-hand side matrix of the linear system.
    f : np.ndarray
        The right-hand side vector of the linear system.
    vtx : np.ndarray
        The array of vertices.
    xlen : float
        The length of the domain in the x direction.
    ylen : float
        The length of the domain in the y direction.

    Returns
    -------
    tuple[sps.coo_matrix, np.ndarray]
        The modified left-hand side matrix and right-hand side vector with
        Dirichlet boundary conditions applied.
    """
    i = interior(vtx, xlen, ylen)
    d = sps.diags(i, dtype=int, format="coo")
    id = sps.identity(len(vtx), dtype=int, format="coo")
    return ((d @ a @ d) + id - d, d @ f)


def assemble_linear_system(
    vtx: np.ndarray,
    elt: np.ndarray,
    xlen: float,
    ylen: float,
    mu: np.ndarray,
    f: np.ndarray,
    m: sps.coo_matrix,
    /,
) -> tuple[sps.coo_matrix, np.ndarray]:
    """
    Assemble the global linear system and apply Dirichlet boundary conditions.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.
    xlen : float
        The length of the domain in the x direction.
    ylen : float
        The length of the domain in the y direction.
    mu : np.ndarray
        An array containing the values of the parameter mu at the vertices.
    f : np.ndarray
        The source term values at the vertices.

    Returns
    -------
    tuple[sps.coo_matrix, np.ndarray]
        The assembled and modified left-hand side matrix and right-hand side vector
        of the linear system with Dirichlet boundary conditions applied.
    """
    lhs = assemble_linear_system_l(vtx, elt, mu)
    rhs = assemble_linear_system_r(m, f)
    return apply_dirichlet_boundary_conditions(lhs, rhs, vtx, xlen, ylen)


def solve(a: sps.coo_matrix, f: np.ndarray, /) -> np.ndarray:
    return sps.linalg.spsolve(a, f)
