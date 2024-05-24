import matplotlib.pyplot as plt
import mesh
import numpy as np
from assembly import assemble_linear_system, mass, solve, stiffness
from functions import h1_norm, l2_norm, mu, source, u_ex

# Globals parameters
ratio = 2
ylen = 5.0
xlen = ylen * ratio
max_ydiv = 75
max_xdiv = max_ydiv * ratio
alphas = [1.0, 2 / 3]


def test_mesh_generation() -> None:
    xlen, ylen = 10.0, 10.0
    for div in [2, 3]:
        vtx, elt = mesh.generate_rectangle_mesh(xlen, ylen, div, div)
        mesh.plot_mesh(vtx, elt, f"L={(xlen, ylen)}, N={(div, div)}")
        vtx, elt = mesh.geometric_refinement(vtx, elt, r=2)
        mesh.plot_mesh(vtx, elt, f"L={(xlen, ylen)}, N={(div, div)}")
    print("test_mesh_generation: success")


def test_mass() -> None:
    xlen, ylen = 10.0, 10.0
    for div in range(1, 10):
        vtx, elt = mesh.generate_rectangle_mesh(xlen, ylen, div, div)
        m = mass(vtx, elt)
        area = np.round(m.sum(), 6)
        if area != xlen * ylen:
            print("test_mass: failure:")
            print(
                f"m.sum() ({area}) should be equal to domain area (i.e xlen * ylen ({xlen * ylen}))"
            )
    print("test_mass: success")


def test_stiffness() -> None:
    xlen, ylen = 10.0, 10.0
    for div in range(1, 10):
        vtx, elt = mesh.generate_rectangle_mesh(xlen, ylen, div, div)
        m = mass(vtx, elt)
        s = stiffness(vtx, elt)
        for _ in range(10):
            alpha1, beta1 = np.random.randn(2), np.random.randn(1)
            alpha2, beta2 = np.random.randn(2), np.random.randn(1)

            u1 = np.dot(vtx, alpha1) + beta1
            u2 = np.dot(vtx, alpha2) + beta2

            n1 = u1.T @ s @ u2
            n2 = np.dot(alpha1, alpha2) * np.sum(m)
            if np.round(n1, 6) != np.round(n2, 6):
                print("test_stiffness: failure:")
                print(
                    f"u1.T @ s @ u2 ({n1}) should be equal to np.dot(alpha1, alpha2) * np.sum(m) ({n2})"
                )
    print("test_stiffness: success")


def compute_error(xdiv: int, ydiv: int, alpha: float, /) -> tuple[float, float]:
    vtx, elt = mesh.generate_rectangle_mesh(xlen, ylen, xdiv, ydiv)

    x, y = vtx[:, 0], vtx[:, 1]

    m = mass(vtx, elt)
    s = stiffness(vtx, elt)

    mu_v = mu(x, y, xlen, ylen)
    f_v = source(x, y, xlen, ylen, alpha)
    a, f = assemble_linear_system(vtx, elt, xlen, ylen, mu_v, f_v, m)

    u_h_v = solve(a, f)
    u_ex_v = u_ex(x, y, xlen, ylen, alpha)

    l2 = l2_norm(u_h_v - u_ex_v, m) / l2_norm(u_ex_v, m)
    h1 = h1_norm(u_h_v - u_ex_v, m, s) / h1_norm(u_ex_v, m, s)
    return l2, h1


def test_convergence() -> None:
    ydivs = np.linspace(5, max_ydiv, num=10, dtype=int)
    xdivs = ydivs * ratio

    l2_norm = np.zeros_like(xdivs, dtype=float)
    h1_norm = np.zeros_like(xdivs, dtype=float)
    for alpha in alphas:
        print("{:=^80s}".format(" Convergence "))
        print(f"a = {alpha:.2f}")
        print(f"L = {(xlen, ylen)}")

        print(f"N = ", end="", flush=True)
        for i, (xdiv, ydiv) in enumerate(zip(xdivs, ydivs)):
            print(f"{(xdiv, ydiv)}", end=" ", flush=True)
            l2, h1 = compute_error(xdiv, ydiv, alpha)
            l2_norm[i] = l2
            h1_norm[i] = h1
        print()

        h = xlen / xdivs
        l2_order = (np.log(l2_norm[0]) - np.log(l2_norm[-1])) / (np.log(h[0]) - np.log(h[-1]))
        h1_order = (np.log(h1_norm[0]) - np.log(h1_norm[-1])) / (np.log(h[0]) - np.log(h[-1]))

        x = np.linspace(h[-1], h[0], num=20)
        plt.figure(figsize=(10, 6))
        plt.plot(h, l2_norm, "o-", label=f"$\mathrm{{L}}^2(\Omega)$ error, order: {l2_order:.2f}")
        plt.plot(h, h1_norm, "s-", label=f"$\mathrm{{H}}^1(\Omega)$ error, order: {h1_order:.2f}")
        plt.plot(x, x, "--", label=r"$O(n)$")
        plt.plot(x, x**2, "--", label=r"$O(n^2)$")

        plt.xlabel("Mesh step $h$")
        plt.ylabel("Error")
        plt.loglog()
        plt.title(rf"Convergence of numerical approximations with $\alpha={alpha:.2f}$")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.grid(True)
        path = f"../resources/convergence-{alpha:.2f}.pdf"
        plt.savefig(path, dpi=300)
        print(f"Figure saved in {path}")
        print("{:=^80s}".format(""))


def test_solution() -> None:
    xdiv, ydiv = max_xdiv, max_ydiv

    for alpha in alphas:
        print("{:=^80s}".format(" Approximation "))
        print(f"a = {alpha:.2f}")
        print(f"L = {(xlen, ylen)}")
        print(f"N = {(xdiv, ydiv)}")
        vtx, elt = mesh.generate_rectangle_mesh(xlen, ylen, xdiv, ydiv)

        x, y = vtx[:, 0], vtx[:, 1]

        m = mass(vtx, elt)

        mu_v = mu(x, y, xlen, ylen)
        f_v = source(x, y, xlen, ylen, alpha)
        a, f = assemble_linear_system(vtx, elt, xlen, ylen, mu_v, f_v, m)

        u_h_v = solve(a, f)
        u_ex_v = u_ex(x, y, xlen, ylen, alpha)

        title = f"L={(xlen, ylen)}, N={(xdiv, ydiv)}"
        func = r"$u_{{h}}$"
        path = f"../resources/approximation-{alpha:.2f}-s.pdf"
        mesh.plot_approximation(vtx, elt, alpha, u_h_v, func, title, path)

        func = r"$u_{{h}} - \Pi_{{h}}u_{{ex}}$"
        path = f"../resources/approximation-{alpha:.2f}-d.pdf"
        mesh.plot_approximation(vtx, elt, alpha, u_h_v - u_ex_v, func, title, path)
        print("{:=^80s}".format(""))


if __name__ == "__main__":
    # test_mesh_generation()
    # test_mass()
    # test_stiffness()
    test_convergence()
    test_solution()
