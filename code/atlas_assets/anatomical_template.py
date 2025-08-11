"""Anatomical template data package management with NIfTI and OME-Zarr support."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr
from ome_zarr.writer import write_multiscale

from allen_atlas_assets.atlas_asset import AtlasAsset
from utils import decompose_affine


@dataclass
class AnatomicalTemplate(AtlasAsset):
    """Anatomical template dataset with multiscale support.

    Attributes:
        scales: Tuple of resolution scales in micrometers per voxel
    """

    scales: tuple

    _asset_location = "anatomical-templates"

    def copy_nifti_files(self, prefix, output_root):
        """Copy NIfTI template files with standardized naming."""
        template_dir = self.location(output_root)
        template_dir.mkdir(parents=True, exist_ok=True)

        for scale in self.scales:
            src = f"{prefix}_{scale}.nii.gz"
            dst_fname = f"anatomical_template_{scale}.nii.gz"
            dst = template_dir / dst_fname
            if not dst.exists():
                shutil.copy2(src, dst)
                logging.info(f"Copied {src} to {dst} with new name")
            else:
                logging.info(f"File {dst} already exists, skipping copy.")

    def convert_nifti_to_omezarr_multiscale(self, output_root):
        """Convert NIfTI files to OME-Zarr multiscale pyramid."""
        input_dir = self.location(output_root)
        output_dir = input_dir
        output_zarr_path = str(
            output_dir / "anatomical_template.ome.zarr"
        )  # zarr expects string path

        logging.info("Starting conversion from NIfTI to OME-Zarr multiscale.")
        logging.info(f"Input directory: {input_dir}")
        logging.info(f"Output Zarr path: {output_zarr_path}")

        arrays = []
        transforms = []
        axes = [
            {"name": "z", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ]

        for scale in self.scales:
            fname = f"anatomical_template_{scale}.nii.gz"
            fpath = input_dir / fname
            logging.info(f"Loading file: {fpath}")
            img = nib.load(str(fpath))
            data = img.get_fdata().astype(np.float32)
            arrays.append(data)
            spacing = img.header.get_zooms()[:3]
            origin = img.affine[:3, 3]
            scale_vec, rotation_mat, translation_vec = decompose_affine(img.affine)
            logging.info(
                f"Scale {scale}: data shape {data.shape}, dtype {data.dtype}, spacing {spacing}, "
                f"origin {origin}, affine:\n{img.affine}\n"
                f"Decomposed: scale={scale_vec}, translation={translation_vec}, rotation=\n{rotation_mat}"
            )
            transforms.append(
                [
                    {"type": "scale", "scale": scale_vec.tolist()},
                ]
            )

        group = zarr.open(output_zarr_path, mode="w")
        logging.info(
            "Writing OME-Zarr multiscale with affine transforms and chunk size (128, 128, 128)..."
        )
        compressor = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}
        write_multiscale(
            arrays,
            group,
            axes=axes,
            coordinate_transformations=transforms,
            chunks=(128, 128, 128),
            compressor=compressor,
        )
        logging.info(
            f"OME-Zarr multiscale with affine transforms written to {output_zarr_path}"
        )

    def create(self, input_prefix: Path, output_root: Path):
        """Create complete anatomical template package with NIfTI and OME-Zarr formats."""
        self.copy_nifti_files(input_prefix, output_root)
        self.convert_nifti_to_omezarr_multiscale(output_root)
        self.create_manifest(output_root)
        logging.info(
            f"Created anatomical template package at {self.location(output_root)}"
        )
