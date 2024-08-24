import numpy as np


def generate_1(n: int) -> np.ndarray:
    return np.fromfunction(lambda i, j: np.power(2, i - j), (n, n), dtype=float)


def generate_2(n: int) -> np.ndarray:
    return np.fromfunction(lambda i, j: 1 / (i + j + 1), (n, n), dtype=float)


def iterate(u, a, n):
    for i in range(n):
        print(f"u{i}: {u}")
        u = a @ u
        u /= np.linalg.norm(u)
    print()


def main() -> None:
    print("---------------- Question 1: ----------------")
    u = np.array([1, 2, 3, 4])
    v = np.array([-1, 0, 1, 2])
    w = np.array([2, -2, 1, 0])

    print(f"u*v^t = {np.outer(u, v)}")
    print("v^t*w = ", np.dot(v, w))
    print("|u|_2 = ", np.linalg.norm(u))
    print("|v|_1 = ", np.linalg.norm(v, 1))
    print("|u-v|_inf = ", np.linalg.norm(u - v, np.infty), "\n")

    print("---------------- Question 2: ----------------")
    print(np.zeros((3, 3), int))
    print(np.eye(3, 3, dtype=int))
    print(np.ones((3, 3), int))
    print(np.diag([1, 2, 3]))

    print("---------------- Question 3: ----------------")
    a = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
    b = np.array([[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]])
    print("A: det:", np.linalg.det(a), "spec:", np.linalg.eigvals(a))
    print("B: det:", np.linalg.det(b), "spec:", np.linalg.eigvals(b))
    c = np.block(np.array([[a, b.T], [b, a]]))
    print("C: det:", np.linalg.det(c))

    print("---------------- Question 4: ----------------")
    print(generate_1(3))
    print(generate_2(3))

    print("---------------- Question 5: ----------------")
    iterate(np.ones(4, dtype=float), a, 20)
    iterate(np.arange(4.0), a, 20)
    iterate(np.random.random(size=4), a, 20)


if __name__ == "__main__":
    main()
