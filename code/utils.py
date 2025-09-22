"""Utility functions shared across atlas processing modules."""

import numpy as np
from typing import List

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

def get_image_orientation(rotation_matrix: np.ndarray,
                          axes_metadata: List,
                          species: str,
                          ):
    
    # Create lookup for orientation. 
    # - Humans (bipeds) use anterior-posterior (front-back) and superior-inferior (head-feet)
    # - Quadrupeds use rostral-caudal (front-back) and dorsal-ventral.

    orientation_lookup = {'AP': ['anterior','posterior'],
                          'DV': ['dorsal', 'ventral'],
                          'LR': ['left', 'right'],
                          'RC': ['rostral', 'caudal'],
                          'SI': ['superior', 'inferior']}