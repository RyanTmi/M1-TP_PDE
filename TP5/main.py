import matplotlib.pyplot as plt
import numpy as np
import pyde_fem as pf
from scipy.sparse import coo_matrix


def plot_sparse_matrix(sm: coo_matrix, name: str) -> None:
    plt.figure(name, figsize=(9, 6))
    plt.imshow(sm.toarray() == 0.0, cmap="gray")
    plt.show()


def test_squared_mesh() -> None:
    vtx = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elt = np.array([[0, 1, 2], [0, 2, 3]])
    eltb = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    m = pf.mass(vtx, elt)
    bm = pf.mass(vtx, eltb)
    s = pf.stiffness(vtx, elt)

    print("=========== basic squarde mesh ===========\n")
    print("Mass matrix")
    print(m.toarray(), "\n")
    print("Boundary mass matrix")
    print(bm.toarray(), "\n")
    print("Stiffness matrix")
    print(s.toarray(), "\n")

    print("Domain Area")
    print(np.sum(m), "\n")  # 1.0
    print("Boundary Length")
    print(np.sum(bm), "\n\n")  # 4.0


def test_mesh6() -> None:
    vtx, elt = pf.mesh.load("assets/mesh6.msh")
    eltb, _ = pf.mesh.boundary(elt)

    m = pf.mass(vtx, elt)
    bm = pf.mass(vtx, eltb)
    s = pf.stiffness(vtx, elt)

    print("=========== mesh6.msh ===========\n")
    print("Domain Area")
    print(np.sum(m), "\n")  # 3.0
    print("Boundary Length")
    print(np.sum(bm), "\n")  # 8.0

    plot_sparse_matrix(m, "Mass matrix")
    plot_sparse_matrix(bm, "Boundary mass matrix")
    plot_sparse_matrix(s, "Stiffness matrix")

    print("|-----------------|-----------------|")
    print("|     U1*K*U2     |  a1*a2*|Omega|  |")
    print("|-----------------|-----------------|")
    for i in range(10):
        alpha1, beta1 = np.random.randn(2), np.random.randn(1)
        alpha2, beta2 = np.random.randn(2), np.random.randn(1)

        u1 = np.dot(vtx, alpha1) + beta1
        u2 = np.dot(vtx, alpha2) + beta2

        print(f"| {u1.T @ s @ u2:15.10f} | {np.dot(alpha1, alpha2) * np.sum(m):15.10f} |")
    print("|-----------------|-----------------|")


def main() -> None:
    test_squared_mesh()
    test_mesh6()


if __name__ == "__main__":
    main()
