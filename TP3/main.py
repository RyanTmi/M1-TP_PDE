import time

import numpy as np
import pyde_fem as pf
import matplotlib.pyplot as plt


def plot_timing_data(times, mesh):
    refine_times = times["refine"]
    boundary_times = times["boundary"]
    connected_component_times = times["connected_component"]
    boundary_connected_component_times = times["boundary_connected_component"]

    x = np.arange(len(refine_times))

    plt.figure()
    plt.plot(x, refine_times, "o-", label="Refine")
    plt.plot(x, boundary_times, "o-", label="Boundary")
    plt.plot(x, connected_component_times, "o-", label="Connected Component")
    plt.plot(x, boundary_connected_component_times, "o-", label="Boundary Connected Component")

    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.title(mesh)
    plt.xticks(x, [str(i) for i in range(len(refine_times))])
    plt.legend()
    plt.grid(True)
    plt.show()


def profile_func(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time


def profile(mesh_file: str) -> None:
    print("\n", mesh_file, sep="")

    (vertices, indices), t = profile_func(pf.mesh.load, mesh_file)
    print(f"Mesh loaded in {t:0.6f} seconds")

    times = {
        "refine": [0.0],
        "boundary": [],
        "connected_component": [],
        "boundary_connected_component": [],
    }
    for i in range(4):
        print(f"\n---------- Refine : {i} ----------\n")

        if i != 0:
            (vertices, indices), t = profile_func(pf.mesh.refine, vertices, indices)
            times["refine"].append(t)
            print(f"Refined x{i} mesh in {t:0.6f} seconds")

        (bi, be2e), t = profile_func(pf.mesh.boundary, indices)
        times["boundary"].append(t)
        print(f"Boundary created in {t:0.6f} seconds")

        _, t = profile_func(pf.mesh.connected_component, indices)
        times["connected_component"].append(t)
        print(f"Connected component created in {t:0.6f} seconds")

        _, t = profile_func(pf.mesh.connected_component, bi)
        times["boundary_connected_component"].append(t)
        print(f"Boundary connected component created in {t:0.6f} seconds")
    print()

    plot_timing_data(times, mesh_file)


def main() -> None:
    mesh_files = [f"assets/mesh{i}.msh" for i in range(1, 7)]

    d = np.random.random(size=2)
    d = d / np.linalg.norm(d)
    for mesh_file in mesh_files:
        print("\n", mesh_file, sep="")
        t1 = time.perf_counter()
        vertices, indices = pf.mesh.load(mesh_file)
        t2 = time.perf_counter()
        print(f"Mesh loaded in {t2 - t1:0.6f} seconds")

        r = 0
        for i in range(r):
            t1 = time.perf_counter()
            vertices, indices = pf.mesh.refine(vertices, indices)
            t2 = time.perf_counter()
            print(f"Refined x{i + 1} mesh in {t2 - t1:0.6f} seconds")

        t1 = time.perf_counter()
        bi, _ = pf.mesh.boundary(indices)
        t2 = time.perf_counter()
        print(f"Boundary created in {t2 - t1:0.6f} seconds")

        t1 = time.perf_counter()
        cc = pf.mesh.connected_component(indices)
        t2 = time.perf_counter()
        print(f"Connected component created in {t2 - t1:0.6f} seconds")

        t1 = time.perf_counter()
        ccb = pf.mesh.connected_component(bi)
        t2 = time.perf_counter()
        print(f"Boundary connected component created in {t2 - t1:0.6f} seconds")

        values = np.cos(4 * np.pi * vertices.dot(d))

        pf.mesh.plot(
            vertices,
            indices,
            boundary_indices=bi,
            connected_components=cc,
            boundary_connected_components=ccb,
            values=values,
        )
        plt.show()


if __name__ == "__main__":
    main()
