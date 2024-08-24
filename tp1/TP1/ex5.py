import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.sin(2 * np.pi * x)


def df(x):
    return 2 * np.pi * np.cos(2 * np.pi * x)


def dfp(x, h):
    return (f(x + h) - f(x)) / h


def dfm(x, h):
    return (f(x) - f(x - h)) / h


def df0(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def main() -> None:
    # -------- Question 1 --------
    h = 0.1
    x = np.linspace(-1, 1, 1000)

    plt.subplot(2, 2, (1, 2))
    plt.plot(x, df(x), label="f'(x)")
    plt.plot(x, dfp(x, h), label="dfp(x)")
    plt.plot(x, dfm(x, h), label="dfm(x)")
    plt.plot(x, df0(x, h), label="df0(x)")
    plt.xlabel("x")
    plt.legend()

    # -------- Question 2 --------
    hs = np.linspace(0.001, 0.1)
    y = np.array([np.abs(df(x) - dfp(x, h)) for h in hs])
    z = np.max(y, axis=1)

    plt.subplot(2, 2, 3)
    plt.plot(hs, z, label="max|f'-dfp|")
    # plt.plot(hs, 20 * hs)
    plt.xlabel("h")
    plt.legend()

    # -------- Question 3 --------
    y = np.array([np.abs(df(x) - df0(x, h)) for h in hs])
    z = np.max(y, axis=1)

    plt.subplot(2, 2, 4)
    plt.plot(hs, z, label="max|f'-df0|")
    # plt.plot(hs, 40 * hs**2)
    plt.xlabel("h")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
