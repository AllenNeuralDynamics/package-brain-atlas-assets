"""Template data package management with NIfTI and OME-Zarr support."""

import logging
import shutil
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path
import nibabel as nib
import numpy as np
import re
import zarr
from ome_zarr.writer import write_multiscale

from atlas_builder.atlas_asset import AtlasAsset
from atlas_builder.coordinate_space import CoordinateSpace
from utils import (
    decompose_affine,
    write_image_orientation,
    correct_coordinate_transforms_rfc5,
    round_transform_values,
)


@dataclass
class Template(AtlasAsset):
    """Template dataset with multiscale support.

    Attributes:
        scales: Tuple of resolution scales in micrometers per voxel
    """

    scales: tuple
    coordinate_space: CoordinateSpace | None = None

    _asset_location: ClassVar[str] = "templates"
    schema_version: ClassVar[str] = "0.1.0"

    @property
    def manifest(self) -> dict:
        m = super().manifest | {
            "scales": list(self.scales),
        }
        if self.coordinate_space is not None:
            m["coordinate_space"] = self.coordinate_space.manifest
        return m

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "Template":
        scales = manifest.get("scales")
        coordinate_space = None
        if "coordinate_space" in manifest:
            coordinate_space = CoordinateSpace.from_manifest(
                manifest["coordinate_space"], root=root
            )
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            scales=tuple(scales),
            coordinate_space=coordinate_space,
        )

    def copy_nifti_files(self, prefix, output_root):
        """Copy NIfTI template files with standardized naming."""
        template_dir = self.location(output_root)
        template_dir.mkdir(parents=True, exist_ok=True)

        for scale in self.scales:
            src = f"{prefix}_{scale}.nii.gz"
            dst_fname = f"template_{scale}.nii.gz"
            dst = template_dir / dst_fname
            logging.info(f"Destination file: {dst}")
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
            output_dir / "template.ome.zarr"
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
            fname = f"template_{scale}.nii.gz"
            fpath = input_dir / fname
            logging.info(f"Loading file: {fpath}")
            img = nib.load(str(fpath))
            data = img.get_fdata().astype(np.float32)
            arrays.append(data)
            spacing = img.header.get_zooms()[:3]
            origin = img.affine[:3, 3]
            scale_vec, rotation_mat, flip_mat, translation_vec = decompose_affine(img.affine)
            scale_vec = round_transform_values(scale_vec, decimals=6)
            translation_vec = round_transform_values(translation_vec, decimals=6)
            rotation_mat = round_transform_values(rotation_mat, decimals=8)
            flip_mat = round_transform_values(flip_mat, decimals=8)
            logging.info(
                f"Scale {scale}: data shape {data.shape}, dtype {data.dtype}, spacing {spacing}, "
                f"origin {origin}, affine:\n{img.affine}\n"
                f"Decomposed: scale={scale_vec}, translation={translation_vec}, rotation=\n{rotation_mat}, "
                f"flip=\n{flip_mat}"
            )
            scale_transforms = []
            if scale_vec is not None:
                scale_transforms.append({"type": "scale", "scale": scale_vec.tolist()})
            if flip_mat is not None:
                flip_mat_affine = np.hstack([flip_mat, np.zeros((3, 1))]).tolist() # Zero padding affine to be of shape (3,4)
                scale_transforms.append({"type": "affine", "affine": flip_mat_affine})
            if rotation_mat is not None:
                scale_transforms.append({"type": "rotation", "rotation": rotation_mat.tolist()})
            if translation_vec is not None:
                scale_transforms.append({"type": "translation", "translation": translation_vec.tolist()})
            transforms.append(scale_transforms)
           
        
        # Update axis info with orientation
        path_str = str(fpath).lower()
        axes_orientation, original_orientation, ax_code = write_image_orientation(img.affine, axes, path_str)
        logging.info(f"Image axis: {ax_code}.\n Axis orientation is set to: {axes_orientation}")

        group = zarr.open(output_zarr_path, mode="w")
        logging.info("Writing OME-Zarr multiscale with affine transforms and chunk size (128, 128, 128)...")
        compressor = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}
        
        write_multiscale(
            arrays,
            group,
            axes=original_orientation,
            coordinate_transformations=transforms,
            chunks=(128, 128, 128),
            compressor=compressor,
        )

        correct_coordinate_transforms_rfc5(group, axes_orientation)

        logging.info(f"OME-Zarr multiscale with affine transforms written to {output_zarr_path}")

    def create(self, input_prefix: Path, output_root: Path):
        """Create complete template package with NIfTI and OME-Zarr formats."""
        self.copy_nifti_files(input_prefix, output_root)
        self.convert_nifti_to_omezarr_multiscale(output_root)
        self.create_manifest(output_root)
        logging.info(
            f"Created template package at {self.location(output_root)}"
        )
