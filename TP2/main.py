import numpy as np
import pyde_fem as pf


def main() -> None:
    mesh_file = 'assets/mesh1.msh'

    vertices, indices = pf.mesh.load(mesh_file)

    d = np.random.random(size=2)
    d = d / np.linalg.norm(d)
    values = np.cos(4 * np.pi * vertices.dot(d))
    pf.mesh.plot(vertices, indices, values=values)


if __name__ == '__main__':
    main()
