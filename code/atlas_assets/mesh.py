from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    vertex_normals: np.ndarray
    face_vertices: np.ndarray
    face_normals: np.ndarray

    @classmethod
    def from_obj(cls, path):
        vertices = []
        vertex_normals = []
        face_vertices = []
        face_normals = []

        with open(path, "r") as obj_file:
            lines = obj_file.read().split("\n")

        for line in lines:
            if line[:2] == "v ":
                vertices.append(line.split()[1:])

            elif line[:3] == "vn ":
                vertex_normals.append(line.split()[1:])

            elif line[:2] == "f ":
                line = line.replace("//", " ").split()[1:]

                face_vertices.append(line[::2])
                face_normals.append(line[1::2])

        vertices = np.array(vertices).astype(float)
        vertex_normals = np.array(vertex_normals).astype(float)
        face_vertices = np.array(face_vertices).astype(int) - 1
        face_normals = np.array(face_normals).astype(int) - 1

        return cls(
            vertices=vertices,
            vertex_normals=vertex_normals,
            face_vertices=face_vertices,
            face_normals=face_normals,
        )
