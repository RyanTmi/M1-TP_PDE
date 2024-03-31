import matplotlib.pyplot as plt
import numpy as np

from ex3 import generate_a_h


def main() -> None:
    for n in range(5, 20, 5):
        a = generate_a_h(n)
        u = np.linalg.solve(a, np.ones(n))
        h = 1 / (n + 1)
        x = [k * h for k in 1 + range(n)]

        plt.subplot(2, 1, 1)
        plt.plot(x, u)

        p = 4
        u = np.linalg.solve(a, [np.sin(p * np.pi * k * h) for k in 1 + range(n)])

        plt.subplot(2, 1, 2)
        plt.plot(x, u)

    plt.show()


if __name__ == '__main__':
    main()
