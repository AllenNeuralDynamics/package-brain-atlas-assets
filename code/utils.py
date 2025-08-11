"""Utility functions shared across atlas processing modules."""

import numpy as np


def decompose_affine(affine):
    """Decompose 4x4 affine matrix into scale, rotation, and translation components."""
    # Extract translation vector from the last column
    translation = affine[:3, 3]

    # Extract 3x3 transformation matrix (top-left block)
    M = affine[:3, :3]

    # Scale: compute the norm of each column vector
    scale = np.linalg.norm(M, axis=0)

    # Rotation: normalize columns to remove scaling
    rotation = M / scale

    return scale, rotation, translation
