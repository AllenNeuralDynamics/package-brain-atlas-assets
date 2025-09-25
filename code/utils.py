"""Utility functions shared across atlas processing modules."""

import numpy as np
from typing import List
import nibabel as nib

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

def write_image_orientation(affine: np.ndarray,
                          axes_metadata: List,
                          path_str: str,
                          ):
    
    # Create lookup for orientation. 
    # - Humans (bipeds) use anterior-posterior (front-back) and superior-inferior (head-feet)
    # - Quadrupeds use rostral-caudal (front-back) and dorsal-ventral.

    if "human" in path_str:
        orientation_start = {'R':'right', 'L':'left', 'P':'anterior', 'A':'posterior', 'I':'superior', 'S':'inferior'}
        orientation_end = {'R':'left',  'L':'right','P':'posterior','A':'anterior','I':'inferior', 'S':'superior'}
    elif "mouse" in path_str:
        orientation_end = {'R':'right', 'L':'left', 'A':'anterior', 'P':'posterior', 'S':'dorsal', 'I':'ventral'}
        orientation_start = {'R':'left',  'L':'right','A':'posterior','P':'anterior','S':'ventral', 'I':'dorsal'}
    else:
        orientation_start = {'R':'right', 'L':'left', 'P':'rostral', 'A':'caudal', 'I':'dorsal', 'S':'ventral'}
        orientation_end = {'R':'left',  'L':'right','P':'caudal','A':'rostral','I':'ventral', 'S':'dorsal'}

    ax_code = nib.aff2axcodes(affine)
    axis_directions = [f"{orientation_start[val]}-to-{orientation_end[val]}" for val in ax_code]
    axis_directions = list(reversed(axis_directions))

    for idx, axis in enumerate(axes_metadata):
        axis.update({"orientation": {"type": "anatomical", "value": axis_directions[idx]}})

    return axes_metadata, ax_code


