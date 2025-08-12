from dataclasses import dataclass

import numpy as np
import logging
import os


@dataclass
class Mesh:
    vertices: np.ndarray
    vertex_normals: np.ndarray
    face_vertices: np.ndarray  # faces (triangles) as indices
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

    @classmethod
    def from_msh(cls, path: str):
        """Load mesh from a binary .msh file produced by Allen tools (moved)."""
        def read_exact(f, n):
            data = f.read(n)
            if len(data) != n:
                raise EOFError("Unexpected end of file")
            return data

        try:
            logging.info("Reading MSH: %s", path)
            with open(path, "rb") as f:
                # Number of points (little-endian uint32)
                num_points = int.from_bytes(read_exact(f, 4), byteorder="little")
                if num_points <= 0:
                    raise ValueError("Invalid number of points in .msh file")
                logging.info("Points: %d", num_points)

                # Read Point3d array: 6 float32 per point (normal[3], coord[3])
                pts_bytes = read_exact(f, num_points * 6 * 4)
                pts = np.frombuffer(pts_bytes, dtype="<f4").reshape(num_points, 6)
                file_vertex_normals = pts[:, 0:3].astype(np.float64)
                vertices = pts[:, 3:6].astype(np.float64)

                # Number of triangle strips (little-endian uint32)
                num_strips = int.from_bytes(read_exact(f, 4), byteorder="little")
                if num_strips < 0:
                    raise ValueError("Invalid number of triangle strips in .msh file")
                logging.info("Triangle strips: %d", num_strips)

                strips = []
                strip_lengths = []
                total_tris = 0
                for _ in range(num_strips):
                    # little-endian uint16 count
                    num_indices = int.from_bytes(read_exact(f, 2), byteorder="little")
                    strip_lengths.append(num_indices)
                    if num_indices == 0:
                        strips.append(np.array([], dtype=np.int64))
                        continue
                    idx_bytes = read_exact(f, num_indices * 4)
                    indices = np.frombuffer(idx_bytes, dtype="<u4").astype(np.int64)
                    strips.append(indices)
                    total_tris += max(0, num_indices - 2)

                if strip_lengths:
                    logging.debug(
                        "Strip lengths: min=%d avg=%.2f max=%d",
                        int(np.min(strip_lengths)),
                        float(np.mean(strip_lengths)),
                        int(np.max(strip_lengths)),
                    )
                logging.info("Triangles (expanded from strips): %d", total_tris)

                # Convert strips to triangle list
                faces = np.empty((total_tris, 3), dtype=np.int32)
                tri_cursor = 0
                for indices in strips:
                    n = len(indices)
                    if n < 3:
                        continue
                    for j in range(n - 2):
                        a, b, c = int(indices[j]), int(indices[j + 1]), int(indices[j + 2])
                        if j % 2 == 1:  # flip winding
                            faces[tri_cursor] = [a, c, b]
                        else:
                            faces[tri_cursor] = [a, b, c]
                        tri_cursor += 1

                # Compute face normals (unit length) for all faces
                if faces.size:
                    v0 = vertices[faces[:, 0]]
                    v1 = vertices[faces[:, 1]]
                    v2 = vertices[faces[:, 2]]
                    face_normals_raw = np.cross(v1 - v0, v2 - v0)
                    fl = np.linalg.norm(face_normals_raw, axis=1)
                    nz = fl > 0
                    degenerate = int((~nz).sum())
                    if degenerate:
                        logging.debug("Degenerate triangles: %d", degenerate)
                    face_normals = np.zeros_like(face_normals_raw)
                    face_normals[nz] = (face_normals_raw[nz].T / fl[nz]).T
                else:
                    face_normals = np.empty((0, 3), dtype=np.float64)

                # Always use vertex normals from file (normalize to unit length)
                file_norm_len = np.linalg.norm(file_vertex_normals, axis=1)
                zero_normals = int((file_norm_len == 0).sum())
                if zero_normals:
                    logging.debug("Vertices with zero-length normals in file: %d", zero_normals)
                vertex_normals = file_vertex_normals.copy()
                nz = file_norm_len > 0
                if np.any(nz):
                    vertex_normals[nz] = (vertex_normals[nz].T / file_norm_len[nz]).T

                logging.info(
                    "Loaded MSH: %d vertices, %d faces", vertices.shape[0], faces.shape[0]
                )

                return cls(
                    vertices=vertices.astype(np.float64),
                    vertex_normals=vertex_normals.astype(np.float32),
                    face_vertices=faces,
                    face_normals=face_normals.astype(np.float32),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh from {path}: {e}")
