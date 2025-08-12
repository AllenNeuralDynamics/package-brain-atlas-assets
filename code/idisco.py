"""Packaging script for iDISCO multichannel anatomical template as OME-Zarr."""

import logging
from pathlib import Path
import numpy as np
import nibabel as nib
import zarr as zarr_lib
import re
from ome_zarr.writer import write_image, write_multiscale
from ome_zarr.io import parse_url
from ome_zarr.format import CurrentFormat
from atlas_assets.anatomical_template import AnatomicalTemplate
from utils import decompose_affine

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
        base = re.sub(r'(_\d+um?|_\d+)?(\.nii(\.gz)?)$', '', base)
        channel_names.append(base)
    if not arrays:
        raise RuntimeError(f"No NIfTI files found in {res_dir}")
    arr = np.stack(arrays, axis=0)
    logging.info(f"Loaded {len(arrays)} channels from {res_dir} with shape {arr.shape}")
    return arr, channel_names


def package_idisco_template(results_dir):
    """Package iDISCO multichannel template as OME-Zarr multiscale pyramid (OME standard)."""
    results_dir = Path(results_dir)
    # Use AnatomicalTemplate location for output_dir
    template = AnatomicalTemplate(
        name="allen-adult-mouse-spim-idisco-template",
        version="2025-05",
        scales=tuple(RESOLUTIONS),
    )
    output_dir = template.location(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "anatomical_template.ome.zarr"
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
        scale_vec, rotation_mat, translation_vec = decompose_affine(img.affine)
        logging.info(
            f"Scale {res}: spacing {spacing}, origin {origin}, affine:\n{img.affine}\n"
            f"Decomposed: scale={scale_vec}, translation={translation_vec}, rotation=\n{rotation_mat}"
        )
        arr, channel_names = load_nifti_channels(res_dir)
        arrays.append(arr)
        if all_channel_names is None:
            all_channel_names = channel_names
        elif channel_names != all_channel_names:
            logging.warning(f"Channel names at {res}um do not match previous scales!")
        coordinate_transformations.append([
            {"type": "scale", "scale": [1.0] + scale_vec.tolist()}
        ])

    if not arrays:
        raise RuntimeError("No valid scales found to write OME-Zarr multiscale.")

    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    compressor = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}
    write_multiscale(
        arrays,
        group,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        chunks=(1, 128, 128, 128),
        compressor=compressor,
        channel_names=all_channel_names,
    )
    logging.info(f"iDISCO OME-Zarr multiscale pyramid written to {zarr_path}")

    # Create and register AnatomicalTemplate asset
    template.create_manifest(results_dir)
    logging.info(f"Created AnatomicalTemplate manifest at {template.location(results_dir)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python idisco.py <output_dir>")
        exit(1)
    package_idisco_template(sys.argv[1])
