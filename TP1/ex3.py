import matplotlib.pyplot as plt
import numpy as np


def generate_a_h(n: int) -> np.ndarray:
    a = -1 * (np.eye(n, n, 1) + np.eye(n, n, -1)) + 2 * np.eye(n, n)
    return a * np.power(n + 1, 2)


def main() -> None:
    d = 5
    a_h = generate_a_h(d)
    ev = [(2 * np.power(d + 1, 2)) * (1 - np.cos(k * np.pi / (d + 1))) for k in 1 + range(d)]
    print(np.linalg.eigvals(a_h))
    print(np.around(ev, decimals=8))

    x = 2 + np.arange(100)
    y = [np.linalg.norm(generate_a_h(k)) for k in x]
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
