import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    x = np.linspace(0, 2 * np.pi)
    for t in np.linspace(0, 4, 5):
        plt.plot(x, np.sin(x - t), label=f"sin(x-{t})")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
