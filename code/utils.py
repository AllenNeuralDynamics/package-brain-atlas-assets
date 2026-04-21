"""Utility functions shared across atlas processing modules."""

import numpy as np
import nibabel as nib
import logging
import copy
import SimpleITK as sitk


def convert_mhd_to_nifti(mhd_path, output_path, output_direction=None, output_origin=None):
    """Convert an MHD file to NIfTI, translating spatial units from microns to millimeters.

    Args:
        mhd_path: Path to the input MHD file.
        output_path: Path where the converted NIfTI file will be written.
        output_direction: Optional direction cosine matrix to assign before writing.
        output_origin: Optional origin in millimeters to assign before writing.
    """
    image = sitk.ReadImage(str(mhd_path))

    spacing_mm = tuple(value / 1000.0 for value in image.GetSpacing())
    image.SetSpacing(spacing_mm)

    origin_mm = tuple(output_origin) if output_origin is not None else tuple(value / 1000.0 for value in image.GetOrigin())
    image.SetOrigin(origin_mm)

    if output_direction is not None:
        image.SetDirection(output_direction)

    sitk.WriteImage(image, str(output_path))


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
        orientation_start = {'L':'right', 'R':'left', 'P':'anterior', 'A':'posterior', 'I':'superior', 'S':'inferior'}
        orientation_end = {'L':'left',  'R':'right','P':'posterior','A':'anterior','I':'inferior', 'S':'superior'}
    elif "mouse" in path_str:
        orientation_start = {'R':'right',  'L':'left','A':'posterior','P':'anterior','S':'ventral', 'I':'dorsal'}
        orientation_end = {'R':'left', 'L':'right', 'A':'anterior', 'P':'posterior', 'S':'dorsal', 'I':'ventral'}
    else:
        orientation_start = {'L':'right', 'R':'left', 'P':'rostral', 'A':'caudal', 'I':'dorsal', 'S':'ventral'}
        orientation_end = {'L':'left',  'R':'right','P':'caudal','A':'rostral','I':'ventral', 'S':'dorsal'}

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

    for idx, axis in enumerate(axes_metadata):
        axis.update({"orientation": {"type": "anatomical", "value": axis_directions[idx]}})
        
    return axes_metadata

def _split_dataset_and_global_transforms(
    coordinate_transformations: list[dict],
) -> tuple[list[dict], list[dict]]:
    dataset_transforms = []
    global_transforms = []
    for transform in coordinate_transformations:
        if transform.get("type") == "scale":
            dataset_transforms.append(copy.deepcopy(transform))
        else:
            global_transforms.append(copy.deepcopy(transform))
    return dataset_transforms, global_transforms


def _transforms_match(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False

    for left_transform, right_transform in zip(left, right):
        if left_transform.get("type") != right_transform.get("type"):
            return False

        left_keys = set(left_transform.keys())
        right_keys = set(right_transform.keys())
        if left_keys != right_keys:
            return False

        for key in left_keys - {"type"}:
            left_value = np.asarray(left_transform[key], dtype=float)
            right_value = np.asarray(right_transform[key], dtype=float)
            if left_value.shape != right_value.shape:
                return False
            if not np.allclose(left_value, right_value, atol=1e-6):
                return False

    return True


def _wrap_transform_sequence(input_name: str, output_name: str, transforms: list[dict]) -> dict:
    return {
        "type": "sequence",
        "input": input_name,
        "output": output_name,
        "transformations": transforms,
    }


def correct_coordinate_transforms_rfc5(
    group,
    axes,
    coordinate_system_name="mm RAS",
    intrinsic_coordinate_system_name="intrinsic",
    multiscale_transform_key="coordinateTransformations",
):
    attrs = dict(group.attrs)
    ome_block = attrs.get("ome")
    if ome_block is None:
        raise ValueError("Expected OME metadata block in group attrs")

    multiscales = ome_block.get("multiscales", [])
    if not multiscales:
        raise ValueError("Expected at least one multiscales entry in OME metadata")

    multiscales_entry = multiscales[0]
    intrinsic_axes = copy.deepcopy(multiscales_entry.get("axes", axes))
    ome_block["coordinateSystems"] = [
        {"name": intrinsic_coordinate_system_name, "axes": intrinsic_axes},
        {"name": coordinate_system_name, "axes": copy.deepcopy(axes)},
    ]
    array_data = multiscales_entry.get("datasets", [])
    global_coordinate_transformations = None

    for idx in range(len(array_data)):
        _array = array_data[idx]

        array_path = _array.get("path", str(idx))

        coord_transforms = _array.get("coordinateTransformations", [])
        dataset_transforms, candidate_global_transforms = _split_dataset_and_global_transforms(coord_transforms)
        if not dataset_transforms:
            raise ValueError(
                f"Dataset {array_path} must include a scale transform to map indices into {intrinsic_coordinate_system_name}"
            )

        if global_coordinate_transformations is None:
            global_coordinate_transformations = candidate_global_transforms
        elif not _transforms_match(global_coordinate_transformations, candidate_global_transforms):
            raise ValueError(
                "All datasets must share the same non-scale transforms to emit a single intrinsic-to-world transform"
            )

        coordinate_transform_metadata = _wrap_transform_sequence(
            array_path,
            intrinsic_coordinate_system_name,
            dataset_transforms,
        )
        _array["coordinateTransformations"] = [coordinate_transform_metadata]

        # Apply same coordinate transform to all zarr arrays
        array_attr = dict(group[array_path].attrs)
        ome_attr = array_attr.get("ome", {})
        ome_attr["coordinateTransformations"] = _array.get("coordinateTransformations")

        logging.info(f"OME attr: {ome_attr}")
        array_attr["ome"] = ome_attr
        group[array_path].attrs.put(array_attr)

    if global_coordinate_transformations:
        multiscales_entry[multiscale_transform_key] = [
            _wrap_transform_sequence(
                intrinsic_coordinate_system_name,
                coordinate_system_name,
                global_coordinate_transformations,
            )
        ]
    else:
        multiscales_entry.pop(multiscale_transform_key, None)

    ome_block["version"] = "0.6"
    ome_block["multiscales"] = [multiscales_entry]
    attrs["ome"] = ome_block
    group.attrs.put(attrs)