"""Utility functions shared across atlas processing modules."""

import numpy as np
import nibabel as nib
import logging
import copy

def decompose_affine(affine: np.ndarray) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Decompose a 4x4 affine matrix into scale, rotation, flip, and translation.

    The linear transform is assumed to be well-approximated by
    ``rotation @ flip @ diag(scale)`` where ``rotation`` is a proper rotation
    matrix, ``flip`` is a diagonal matrix with entries in ``{-1, 1}``, and
    ``scale`` contains positive axis scales.
    """
    affine = np.asarray(affine, dtype=float)

    # Extract translation vector from the last column
    translation = affine[:3, 3]

    # Extract 3x3 transformation matrix (top-left block)
    M = affine[:3, :3]

    if np.allclose(M, 0.0):
        raise ValueError("Affine linear transform is singular and cannot be decomposed")

    # Estimate the closest orthogonal matrix, then enforce det(rotation) == 1.
    U, _, Vt = np.linalg.svd(M)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1
        rotation = U @ Vt

    # Separate signed per-axis scaling after removing the proper rotation.
    aligned = rotation.T @ M
    signed_scale = np.diag(aligned)
    flip_diag = np.where(signed_scale < 0, -1.0, 1.0)
    scale = np.abs(signed_scale)
    flip = np.diag(flip_diag)

    residual = aligned - np.diag(signed_scale)
    if not np.allclose(residual, 0.0, atol=1e-6):
        logging.warning(
            "Affine contains shear or non-axis-aligned scaling; decompose_affine is approximating it as rotation + flip + scale. Residual:\n%s",
            residual,
        )

    # Replace no-op components with None
    scale_out = None if np.allclose(scale, np.ones_like(scale)) else scale
    rotation_out = None if np.allclose(rotation, np.eye(3)) else rotation
    flip_out = None if np.allclose(flip, np.eye(3)) else flip
    translation_out = None if np.allclose(translation, np.zeros_like(translation)) else translation
    
    return scale_out, rotation_out, flip_out, translation_out


def round_transform_values(
    values: np.ndarray | list[float] | None,
    decimals: int = 6,
) -> np.ndarray | None:
    """Round transform values to avoid floating point precision artifacts."""
    if values is None:
        return None
    arr = np.array(values, dtype=float)
    arr = np.round(arr, decimals=decimals)
    arr = np.where(np.isclose(arr, 0.0), 0.0, arr)
    return arr

def write_image_orientation(affine: np.ndarray,
                          axes_metadata: list,
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