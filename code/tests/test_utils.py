import math
import unittest

import numpy as np

from utils import decompose_affine


def rotation_x(theta: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ]
    )


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ]
    )


def rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def compose_affine(
    scale: np.ndarray | None,
    rotation: np.ndarray | None,
    flip: np.ndarray | None,
    translation: np.ndarray | None,
) -> np.ndarray:
    linear = np.eye(3)
    if rotation is not None:
        linear = rotation @ linear
    if flip is not None:
        linear = linear @ flip
    if scale is not None:
        linear = linear @ np.diag(scale)

    affine = np.eye(4)
    affine[:3, :3] = linear
    if translation is not None:
        affine[:3, 3] = translation
    return affine


class DecomposeAffineTests(unittest.TestCase):
    def assert_recomposes(self, affine: np.ndarray) -> None:
        scale, rotation, flip, translation = decompose_affine(affine)
        recomposed = compose_affine(scale, rotation, flip, translation)

        self.assertTrue(np.allclose(recomposed, affine, atol=1e-6))
        if rotation is not None:
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=6)
        if flip is not None:
            self.assertTrue(np.allclose(flip, np.diag(np.diag(flip)), atol=1e-6))
            self.assertTrue(np.all(np.isin(np.diag(flip), [-1.0, 1.0])))

    def test_recomposes_identity_affine(self) -> None:
        self.assert_recomposes(np.eye(4))

    def test_recomposes_rotation_flip_scale_translation_affine(self) -> None:
        rotation = rotation_z(math.radians(20.0)) @ rotation_y(math.radians(-35.0)) @ rotation_x(math.radians(15.0))
        flip = np.diag([-1.0, 1.0, -1.0])
        scale = np.array([0.025, 0.040, 0.050])
        translation = np.array([1.2, -3.4, 5.6])

        affine = np.eye(4)
        affine[:3, :3] = rotation @ flip @ np.diag(scale)
        affine[:3, 3] = translation

        self.assert_recomposes(affine)

    def test_recomposes_flip_and_scale_only_affine(self) -> None:
        affine = np.eye(4)
        affine[:3, :3] = np.diag([-0.01, 0.02, -0.03])
        affine[:3, 3] = np.array([-2.0, 4.5, 0.0])

        self.assert_recomposes(affine)


if __name__ == "__main__":
    unittest.main()