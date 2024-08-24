import matplotlib.pyplot as plt
import numpy as np
import pyde_fem as pf
import scipy
import scipy.linalg


def main() -> None:
    vtx, elt = pf.mesh.generate(10, 10, 2 * np.pi, np.pi, "assets/rectangle.msh")
    belt, _ = pf.mesh.boundary(elt)
    bn = pf.mesh.boundary_normals(vtx, belt)

    pf.mesh.plot(vtx, elt, boundary_indices=belt, boundary_normals=bn)
    plt.show()
    return
    d = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    x_c = np.array([np.pi, np.pi / 2])
    mu = 2
    u_ex = np.sinh(mu * np.dot(vtx - x_c, d))

    f_ex = np.outer(d, mu * np.cosh(mu * np.dot(vtx[belt][:, 0, :] - x_c, d))).T

    print(f_ex.shape)
    print(bn.shape)
    print(pf.mass(vtx, belt).shape)
    # return
    a = pf.stiffness(vtx, elt) - mu * mu * pf.mass(vtx, elt)
    f = pf.mass(vtx, belt) * np.dot(f_ex, bn)
    u_h = scipy.linalg.solve(a, f)

    pf.mesh.plot(vtx, elt, boundary_indices=belt, boundary_normals=bn, values=u_ex)
    plt.show()
    pf.mesh.plot(vtx, elt, boundary_indices=belt, boundary_normals=bn, values=u_h)
    plt.show()


if __name__ == "__main__":
    main()
