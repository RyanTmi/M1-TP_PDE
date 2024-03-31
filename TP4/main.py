import numpy as np
import matplotlib.pyplot as plt


def create_k(n: int) -> np.ndarray:
    k = np.diag(2 * np.ones(n + 1)) - np.diag(np.ones(n), k=1) - np.diag(np.ones(n), k=-1)
    k[0, 0] = k[n, n] = 1
    return n * k


def create_m(n: int) -> np.ndarray:
    m = np.diag(4 * np.ones(n + 1)) + np.diag(np.ones(n), k=1) + np.diag(np.ones(n), k=-1)
    m[0, 0] = m[n, n] = 2
    return m / (6 * n)


def main() -> None:
    p, mu = 3, 1

    for n in np.linspace(10, 1000, num=10, dtype=int):
        x = np.linspace(0, 1, num=n + 1)

        nv = np.cos(p * np.pi * x)

        k = create_k(n)
        m = create_m(n)
        a = k + mu * mu * m
        f = np.dot(m, nv)

        u = np.linalg.solve(a, f)
        plt.plot(x, u)

    plt.show()


if __name__ == '__main__':
    main()
