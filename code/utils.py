"""Utility functions shared across atlas processing modules."""

import numpy as np
import nibabel as nib
import logging
import copy
import zarr
import SimpleITK as sitk

from dataclasses import asdict

from ngff_zarr import GLASBEY_COLORS
from ngff_zarr.v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    AnatomicalOrientation,
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Dataset,
    Identity,
    Metadata,
    Rotation,
    Scale,
    TransformSequence,
    Translation,
)

# The 0.6rc0 spec document uses "0.6rc0", but no reader accepts that string. ngff-zarr
# accepts "0.6" and "0.6.dev4"; the latter is temporary scaffolding slated for removal
# once 0.6 is final (fideus-labs/ngff-zarr#561, #565).
NGFF_VERSION = "0.6"


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
        orientation_start = {'R':'left',  'L':'right','A':'posterior','P':'anterior','S':'ventral', 'I':'dorsal'}
        orientation_end = {'R':'right', 'L':'left', 'A':'anterior', 'P':'posterior', 'S':'dorsal', 'I':'ventral'}
    else:
        orientation_start = {'L':'right', 'R':'left', 'P':'rostral', 'A':'caudal', 'I':'dorsal', 'S':'ventral'}
        orientation_end = {'L':'left',  'R':'right','P':'caudal','A':'rostral','I':'ventral', 'S':'dorsal'}

    updated_axis = copy.deepcopy(axes_metadata)

    # Original axes
    ax_code_orig = ['S','A','R'] #Default for identity matrix in Nibabel
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


def _strip_none(obj):
    """Recursively drop None-valued keys.

    In OME-Zarr an absent field and an explicit null mean the same thing, so culling
    None keeps the written metadata clean without enumerating optional fields. Only
    None is removed; 0, "" and empty collections are preserved.
    """
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj


def _axis_from_dict(axis: dict) -> Axis:
    """Build an ngff-zarr Axis from the plain dict form used throughout this module."""
    orientation = axis.get("orientation")
    return Axis(
        name=axis["name"],
        type=axis.get("type"),
        unit=axis.get("unit"),
        orientation=(
            AnatomicalOrientation(value=orientation["value"]) if orientation else None
        ),
    )


def _transform_from_dict(transform: dict):
    """Build an ngff-zarr transform from the plain dict form produced by decompose_affine."""
    transform_type = transform.get("type")
    if transform_type == "scale":
        return Scale(scale=list(transform["scale"]))
    if transform_type == "translation":
        return Translation(translation=list(transform["translation"]))
    if transform_type == "rotation":
        return Rotation(rotation=[list(row) for row in transform["rotation"]])
    if transform_type == "affine":
        return Affine(affine=[list(row) for row in transform["affine"]])
    if transform_type == "identity":
        return Identity()
    raise ValueError(f"Unsupported coordinate transformation type: {transform_type!r}")


def _wrap_transform_sequence(input_ref, output_ref, transforms: list[dict]) -> TransformSequence:
    return TransformSequence(
        input=input_ref,
        output=output_ref,
        transformations=[_transform_from_dict(t) for t in transforms],
    )


def omero_from_channel_names(channel_names, array=None):
    """Build omero rendering metadata for named channels.

    Replaces the omero block ome-zarr's write_multiscale derived from channel_names.
    Window bounds come from the array's dtype range when one is supplied.
    """
    if not channel_names:
        return None

    if array is not None and np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        low, high = float(info.min), float(info.max)
    else:
        low, high = 0.0, 1.0

    return Omero(
        channels=[
            OmeroChannel(
                color=GLASBEY_COLORS[i % len(GLASBEY_COLORS)].lstrip("#"),
                window=OmeroWindow(min=low, max=high, start=low, end=high),
                label=label,
            )
            for i, label in enumerate(channel_names)
        ]
    )


def write_multiscale_arrays(
    group,
    arrays,
    dataset_paths=None,
    chunks=(128, 128, 128),
    compressor=None,
):
    """Write pyramid levels into a zarr group, returning the paths written.

    Only the arrays are written; multiscale metadata is written separately by
    write_v06_metadata. Levels are supplied pre-computed, so nothing is downsampled here.
    """
    if compressor is None:
        compressor = zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        )
    if dataset_paths is None:
        dataset_paths = [f"s{i}" for i in range(len(arrays))]

    for path, array in zip(dataset_paths, arrays):
        group.create_array(
            path,
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks,
            compressors=(compressor,),
        )[...] = array
        logging.info(f"Wrote {path}: shape {array.shape}, dtype {array.dtype}")

    return list(dataset_paths)


def write_v06_metadata(
    group,
    dataset_paths,
    coordinate_transformations,
    intrinsic_axes,
    world_axes=None,
    world_coordinate_system_name="mm RAS",
    intrinsic_coordinate_system_name="intrinsic",
    name="/",
    omero=None,
):
    """Write OME-Zarr v0.6 multiscales metadata for arrays already present in ``group``.

    Each entry of ``coordinate_transformations`` holds the transforms for one pyramid
    level, as produced by decompose_affine. The ``scale`` transforms are per-level and
    map array indices into the intrinsic coordinate system; every other transform is
    shared across levels and describes intrinsic -> world, so it is emitted once.
    """
    if len(dataset_paths) != len(coordinate_transformations):
        raise ValueError(
            f"Got {len(dataset_paths)} dataset paths but "
            f"{len(coordinate_transformations)} transform lists"
        )
    if world_axes is None:
        world_axes = intrinsic_axes

    shared_transforms = None
    datasets = []

    for array_path, level_transforms in zip(dataset_paths, coordinate_transformations):
        level_scale_transforms, candidate_shared = _split_dataset_and_global_transforms(level_transforms)
        if not level_scale_transforms:
            raise ValueError(
                f"Dataset {array_path} must include a scale transform to map indices into {intrinsic_coordinate_system_name}"
            )

        if shared_transforms is None:
            shared_transforms = candidate_shared
        elif not _transforms_match(shared_transforms, candidate_shared):
            raise ValueError(
                "All datasets must share the same non-scale transforms to emit a single intrinsic-to-world transform"
            )

        datasets.append(
            Dataset(
                path=array_path,
                coordinateTransformations=[
                    _wrap_transform_sequence(
                        CoordinateSystemIdentifier(path=array_path),
                        CoordinateSystemIdentifier(name=intrinsic_coordinate_system_name),
                        level_scale_transforms,
                    )
                ],
            )
        )

    # Build permutation to swap first and last spatial axes
    # so the intrinsic order (z=R, y=A, x=S) maps to mm RAS (z=S, y=A, x=R)
    spatial_indices = [i for i, a in enumerate(world_axes) if a.get("type") == "space"]
    if len(spatial_indices) >= 2:
        n = len(world_axes)
        perm = np.eye(n, n + 1)
        first, last = spatial_indices[0], spatial_indices[-1]
        perm[[first, last]] = perm[[last, first]]
        if not np.allclose(perm[:, :n], np.eye(n)):
            if shared_transforms is None:
                shared_transforms = []
            shared_transforms.append({"type": "affine", "affine": perm.tolist()})

    metadata = Metadata(
        coordinateSystems=[
            CoordinateSystem(
                name=intrinsic_coordinate_system_name,
                axes=[_axis_from_dict(a) for a in intrinsic_axes],
            ),
            CoordinateSystem(
                name=world_coordinate_system_name,
                axes=[_axis_from_dict(a) for a in world_axes],
            ),
        ],
        datasets=datasets,
        coordinateTransformations=(
            [
                _wrap_transform_sequence(
                    CoordinateSystemIdentifier(name=intrinsic_coordinate_system_name),
                    CoordinateSystemIdentifier(name=world_coordinate_system_name),
                    shared_transforms,
                )
            ]
            if shared_transforms
            else None
        ),
        omero=omero,
        name=name,
    )

    # Metadata.extra is a read-side validation aid, never part of a written entry.
    metadata_dict = asdict(metadata)
    metadata_dict.pop("extra", None)

    attrs = dict(group.attrs)
    attrs["ome"] = {
        "version": NGFF_VERSION,
        "multiscales": [_strip_none(metadata_dict)],
    }
    group.attrs.put(attrs)
    logging.info(
        f"Wrote OME-Zarr {NGFF_VERSION} metadata: {len(datasets)} levels, "
        f"{intrinsic_coordinate_system_name} -> {world_coordinate_system_name}"
    )