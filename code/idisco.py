"""Packaging script for iDISCO multichannel anatomical template as OME-Zarr."""

import logging
from pathlib import Path
import numpy as np
import nibabel as nib
import zarr as zarr_lib
import re
from ngff_zarr import compute_omero_from_ngff_image, to_ngff_image
from atlas_builder.template import Template
from utils import (
    decompose_affine,
    write_multiscale_arrays,
    write_v06_metadata,
)

# Directory containing the multiresolution, multichannel NIfTI files
IDISCO_DATA_DIR = Path("/root/capsule/data/idisco_template_multichannel_multiresolution")
RESOLUTIONS = [10, 25, 50, 100]  # microns


def load_nifti_channel(file_path):
    """Load a single NIfTI file as a float32 array."""
    img = nib.load(str(file_path))
    arr = img.get_fdata(dtype=np.float32)
    return arr


def load_nifti_channels(res_dir):
    """Load all NIfTI files in a resolution directory as a list of arrays."""
    niftis = sorted(res_dir.glob("*.nii*"))
    arrays = []
    channel_names = []
    for f in niftis:
        logging.info(f"Loading NIfTI file: {f}")
        img = nib.load(str(f))
        arrays.append(img.get_fdata(dtype=np.float32))
        # Remove the last _<digits>um or _<digits> before the extension
        base = f.name
        base = re.sub(r"(_\d+um?|_\d+)?(\.nii(\.gz)?)$", "", base)
        channel_names.append(base)
    if not arrays:
        raise RuntimeError(f"No NIfTI files found in {res_dir}")
    arr = np.stack(arrays, axis=0)
    logging.info(f"Loaded {len(arrays)} channels from {res_dir} with shape {arr.shape}")
    return arr, channel_names


def package_idisco_template(results_dir):
    """Package iDISCO multichannel template as OME-Zarr multiscale pyramid (OME standard)."""
    results_dir = Path(results_dir)
    # Use Template location for output_dir
    template = Template(
        name="allen-adult-mouse-spim-idisco-template",
        version="2025-05",
        scales=tuple(RESOLUTIONS),
    )
    output_dir = template.location(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "template.ome.zarr"
    group = zarr_lib.open(str(zarr_path), mode="w")

    arrays = []
    all_channel_names = None
    coordinate_transformations = []
    for res in RESOLUTIONS:
        res_dir = IDISCO_DATA_DIR / str(res)
        if not res_dir.exists():
            logging.warning(f"Resolution dir {res_dir} missing, skipping.")
            continue
        niftis = sorted(res_dir.glob("*.nii*"))
        if not niftis:
            logging.warning(f"No NIfTI files found in {res_dir}, skipping.")
            continue
        # Load first file to get affine and spacing
        img = nib.load(str(niftis[0]))
        spacing = img.header.get_zooms()[:3]
        origin = img.affine[:3, 3]
        scale_vec, rotation_mat, flip_mat, translation_vec = decompose_affine(img.affine)
        logging.info(
            f"Scale {res}: spacing {spacing}, origin {origin}, affine:\n{img.affine}\n"
            f"Decomposed: scale={scale_vec}, translation={translation_vec}, rotation=\n{rotation_mat}, "
            f"flip=\n{flip_mat}"
        )
        arr, channel_names = load_nifti_channels(res_dir)
        arrays.append(arr)
        if all_channel_names is None:
            all_channel_names = channel_names
        elif channel_names != all_channel_names:
            logging.warning(f"Channel names at {res}um do not match previous scales!")

        per_scale_transforms = []
        if scale_vec is not None:
            per_scale_transforms.append({"type": "scale", "scale": [1.0] + scale_vec.tolist()})
        if flip_mat is not None:
            f = np.identity(4)
            f[1:4,1:4] = flip_mat
            per_scale_transforms.append({"type": "affine", "affine": f.tolist()})
        if rotation_mat is not None:
            r = np.identity(4)
            r[1:4,1:4] = rotation_mat
            per_scale_transforms.append({"type": "rotation", "rotation": r.tolist()})
        if translation_vec is not None:
            per_scale_transforms.append({"type": "translation", "translation": [1.0] + translation_vec.tolist()})
        coordinate_transformations.append(per_scale_transforms)

    if not arrays:
        raise RuntimeError("No valid scales found to write OME-Zarr multiscale.")

    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    dataset_paths = write_multiscale_arrays(group, arrays, chunks=(1, 128, 128, 128))

    # Channel rendering metadata: per-channel data range and a 2%-98% display window.
    # Computed from the coarsest level, which tracks the full-resolution distribution
    # closely enough for a display window at a fraction of the cost.
    omero = compute_omero_from_ngff_image(
        to_ngff_image(arrays[-1], dims=["c", "z", "y", "x"]),
        labels=all_channel_names,
    )

    write_v06_metadata(
        group,
        dataset_paths,
        coordinate_transformations,
        intrinsic_axes=axes,
        world_coordinate_system_name="micrometer RAS",
        omero=omero,
    )
    logging.info(f"iDISCO OME-Zarr multiscale pyramid written to {zarr_path}")

    # Create and register Template asset
    template.create_manifest(results_dir)
    logging.info(f"Created Template manifest at {template.location(results_dir)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python idisco.py <output_dir>")
        exit(1)
    package_idisco_template(sys.argv[1])
