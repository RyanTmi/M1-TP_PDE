import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
sns.set_theme()


def generate_rectangle_mesh(
    xlen: float, ylen: float, xdiv: int, ydiv: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a rectangular mesh.

    Parameters
    ----------
    xlen : float
        The length of the rectangle in the x direction.
    ylen : float
        The length of the rectangle in the y direction.
    xdiv : int
        The number of divisions along the x axis.
    ydiv : int
        The number of divisions along the y axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the vertices and elements of the mesh.
    """
    x_vtx = np.linspace(0, xlen, xdiv + 1)
    y_vtx = np.linspace(0, ylen, ydiv + 1)
    xx, yy = np.meshgrid(x_vtx, y_vtx)

    vtx = np.array([xx.ravel(), yy.ravel()]).T
    elt = np.zeros((2 * xdiv * ydiv, 3), dtype=int)

    i = 0
    for u in range(ydiv):
        y = (1 + xdiv) * u
        for x in range(y, xdiv + y):
            i0, i1, i2, i3 = x, x + 1, x + xdiv + 2, x + xdiv + 1
            elt[i : i + 2] = [[i0, i1, i2], [i0, i2, i3]]
            i += 2

    return vtx, elt


def geometric_refinement(
    vtx: np.ndarray, elt: np.ndarray, /, *, r: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform geometric refinement of a mesh.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.
    r : int, optional
        The refinement level. Defaults to 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the refined vertices and elements.

    Raises
    ------
    AssertionError
        If r is less than 1.
    """
    assert r >= 1, "r must be >= 1"

    e1, e2 = elt[0:2]
    v1, v2 = vtx[e1], vtx[e2]

    xi = len(vtx) + np.arange(3)
    yi = np.hstack([e1, e2[2]])
    new_vtx = np.array([v1[1], v1[2], v2[2]]) / 2.0
    new_elt = np.array(
        [
            [yi[0], xi[0], xi[1]],
            [yi[0], xi[1], xi[2]],
            [xi[0], yi[1], xi[1]],
            [xi[1], yi[1], yi[2]],
            [xi[1], yi[2], yi[3]],
            [xi[2], xi[1], yi[3]],
        ]
    )

    rvtx, relt = np.vstack([vtx, new_vtx]), np.vstack([new_elt, elt[2:]])
    return (rvtx, relt) if r == 1 else geometric_refinement(rvtx, relt, r=r - 1)


def interior(vtx: np.ndarray, xlen: float, ylen: float, /) -> np.ndarray:
    """
    Determine the interior points of the mesh.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    xlen : float
        The length of the rectangle in the x direction.
    ylen : float
        The length of the rectangle in the y direction.

    Returns
    -------
    np.ndarray
        A binary array indicating interior points
        (i.e interior[i] == 0 if vtx[i] is on the boundary).
    """
    x = vtx[:, 0]
    y = vtx[:, 1]
    i = (x > 0) & (x < xlen) & (y > 0) & (y < ylen)
    return i.astype(int)


def plot_mesh(vtx: np.ndarray, elt: np.ndarray, title: str, /):
    """
    Plot the mesh.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot()

    trig = mtri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
    ax.triplot(trig, label="Elements")

    ax.set_title(f"Mesh Visualization for {title}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.margins(0.2)
    ax.legend()

    plt.show()


def plot_approximation(
    vtx: np.ndarray,
    elt: np.ndarray,
    alpha: float,
    values: np.ndarray,
    func: str,
    title: str,
    path: None | str = None,
    /,
):
    """
    Plot the function approximation.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices.
    elt : np.ndarray
        The array of elements.
    alpha : float
        The alpha parameter for the function.
    values : np.ndarray
        The values of the function at the vertices.
    func : str
        The name of the function being approximated.
    title : str
        The title of the plot.
    path : str or None, optional
        The path to save the plot. If None, the plot is displayed.
        Defaults to None.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot()

    trig = mtri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
    m = ax.tripcolor(trig, values, label=f"({func})$(x, y)$", cmap="viridis")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(m, cax=cax)

    figure_title = rf"{func} approximation with $\alpha={alpha:.2f}$ for {title}"
    ax.set_title(figure_title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.legend()

    if path is None:
        plt.show()
    else:
        print(f"Figure saved in {path}")
        fig.savefig(path, dpi=300)
