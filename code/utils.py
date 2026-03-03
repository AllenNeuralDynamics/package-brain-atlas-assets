"""Utility functions shared across atlas processing modules."""

import numpy as np
from typing import List
import nibabel as nib
import logging
import copy

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

    # Replace no-op components with None
    scale_out = None if np.allclose(scale, np.ones_like(scale)) else scale
    rotation_out = None if np.allclose(rotation, np.eye(3)) else rotation
    translation_out = None if np.allclose(translation, np.zeros_like(translation)) else translation
    
    return scale_out, rotation_out, translation_out


def round_transform_values(values, decimals=6):
    """Round transform values to avoid floating point precision artifacts."""
    if values is None:
        return None
    arr = np.array(values, dtype=float)
    arr = np.round(arr, decimals=decimals)
    arr = np.where(np.isclose(arr, 0.0), 0.0, arr)
    return arr

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
        orientation_start = {'R':'right',  'L':'left','A':'posterior','P':'anterior','S':'ventral', 'I':'dorsal'}
        orientation_end = {'R':'left', 'L':'right', 'A':'anterior', 'P':'posterior', 'S':'dorsal', 'I':'ventral'}
    else:
        orientation_start = {'R':'right', 'L':'left', 'P':'rostral', 'A':'caudal', 'I':'dorsal', 'S':'ventral'}
        orientation_end = {'R':'left',  'L':'right','P':'caudal','A':'rostral','I':'ventral', 'S':'dorsal'}

    updated_axis = copy.deepcopy(axes_metadata)

    # Original axes
    ax_code_orig = ['R','A','S'] #Default for identity matrix in Nibabel
    axes_metadata = _update_axis_code(axes_metadata, ax_code_orig, orientation_start, orientation_end)

    # Rotated/Transformed axes
    ax_code = nib.aff2axcodes(affine)
    updated_axis = _update_axis_code(updated_axis, ax_code, orientation_start, orientation_end)

    return axes_metadata, updated_axis, ax_code

def _update_axis_code(axes_metadata, ax_code, orientation_start, orientation_end):

    axis_directions = [f"{orientation_start[val]}-to-{orientation_end[val]}" for val in ax_code]
    axis_directions = list(reversed(axis_directions))

    for idx, axis in enumerate(axes_metadata):
        axis.update({"orientation": {"type": "anatomical", "value": axis_directions[idx]}})
        
    return axes_metadata

def correct_coordinate_transforms_rfc5(group, axes, coordinate_system_name="mm"):
    attrs = dict(group.attrs)
    ome_block = attrs.get("ome")
    ome_block["coordinateSystems"] = [
        {"name": coordinate_system_name, "axes": axes}
    ]
    multiscales = ome_block.get("multiscales", [])[0]
    array_data = multiscales.get("datasets", []) 
    for idx in range(len(array_data)):
        _array = array_data[idx]

        array_path = _array.get("path", str(idx))

        # this is being written as a list of transformations. 
        # for RFC5, we want to save a "sequence" of transformations
        coord_transforms = _array.get("coordinateTransformations", [])
        coordinate_transform_metadata = {
            "type": "sequence",
            "input": array_path,
            "output": "mm",
            "transformations": coord_transforms
        }
        _array["coordinateTransformations"] = [coordinate_transform_metadata]

        # Apply same coordinate transform to all zarr arrays
        array_attr = group[array_path].attrs
        ome_attr = array_attr.get("ome", {})
        ome_attr["coordinateTransformations"] = _array.get("coordinateTransformations")
        
        logging.info(f"OME attr: {ome_attr}")
        array_attr["ome"] = ome_attr
        group[array_path].attrs.put(array_attr)

    ome_block["multiscales"] = [multiscales]
    attrs["ome"] = ome_block
    group.attrs.put(attrs)