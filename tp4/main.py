import matplotlib.pyplot as plt
import numpy as np


def create_k(n: int) -> np.ndarray:
    k = np.diag(2 * np.ones(n + 1)) - np.diag(np.ones(n), k=1) - np.diag(np.ones(n), k=-1)
    k[0, 0] = k[n, n] = 1
    return n * k


def create_m(n: int) -> np.ndarray:
    m = np.diag(4 * np.ones(n + 1)) + np.diag(np.ones(n), k=1) + np.diag(np.ones(n), k=-1)
    m[0, 0] = m[n, n] = 2
    return m / (6 * n)


def solve(p: int, mu: float) -> None:
    plt.figure("Solution", figsize=(9, 6))
    for n in np.linspace(10, 1000, num=10, dtype=int):
        x = np.linspace(0, 1, num=n + 1)

        nv = np.cos(p * np.pi * x)

        k = create_k(n)
        m = create_m(n)
        a = k + mu * mu * m
        f = np.dot(m, nv)

        u = np.linalg.solve(a, f)
        plt.plot(x, u)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def error(p: int, mu: float) -> None:
    h = np.zeros((10,))
    y = np.zeros((10,))
    for i, n in enumerate(np.linspace(10, 1000, num=10, dtype=int)):
        x = np.linspace(0, 1, num=n + 1)
        k = create_k(n)
        m = create_m(n)
        a = k + mu * mu * m

        nv = np.cos(p * np.pi * x)
        f = np.dot(m, nv)
        u = np.linalg.solve(a, f)
        ru = np.cos(p * np.pi * x) / (p**2 * np.pi**2 + 1)

        v = ru - u
        h[i] = 1 / n
        y[i] = np.sqrt(v.T @ m @ v) / np.sqrt(u.T @ m @ u)

    plt.figure("Error", figsize=(9, 6))
    plt.xlabel("h")
    plt.ylabel("error")
    plt.loglog()
    plt.plot(h, y)
    plt.show()


def dirichlet(p: int, mu: float):
    plt.figure("Dirichlet solution", figsize=(9, 6))
    for n in np.linspace(10, 1000, num=10, dtype=int):
        x = np.linspace(0, 1, num=n + 1)

        nv = np.sin(p * np.pi * x)

        k = create_k(n)
        m = create_m(n)
        a = k + mu * mu * m
        f = np.dot(m, nv)

        a[1, 1:] = a[-2, :-2] = 0
        f[0] = f[-1] = 0
        u = np.linalg.solve(a, f)
        plt.plot(x, u)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def error_dirichlet(p: int, mu: float) -> None:
    h = np.zeros((10,))
    y = np.zeros((10,))
    for i, n in enumerate(np.linspace(10, 1000, num=10, dtype=int)):
        x = np.linspace(0, 1, num=n + 1)
        k = create_k(n)
        m = create_m(n)
        a = k + mu * mu * m

        nv = np.sin(p * np.pi * x)
        f = np.dot(m, nv)

        a[1, 1:] = a[-2, :-2] = 0
        f[0] = f[-1] = 0
        u = np.linalg.solve(a, f)
        ru = np.sin(p * np.pi * x) / (p**2 * np.pi**2 + 1)

        v = ru - u
        h[i] = 1 / n
        y[i] = np.sqrt(v.T @ m @ v) / np.sqrt(u.T @ m @ u)

    plt.figure("Error", figsize=(9, 6))
    plt.xlabel("h")
    plt.ylabel("error")
    plt.loglog()
    plt.plot(h, y)
    plt.show()


def main() -> None:
    p, mu = 3, 1

    solve(p, mu)
    error(p, mu)

    dirichlet(p, mu)
    error_dirichlet(p, mu)


if __name__ == "__main__":
    main()
