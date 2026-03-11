"""Brain structure annotation management with multiscale OME-Zarr support."""

import logging
import shutil
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tensorstore as ts
import zarr


from ome_zarr.writer import write_multiscale, write_multiscales_metadata
import SimpleITK as sitk

from atlas_builder.template import Template
from atlas_builder.atlas_asset import AtlasAsset
from atlas_builder.terminology import Terminology
from atlas_builder.precomputed import (convert_compressed_annotations_to_precomputed,
                                      write_segment_properties, create_mesh_from_compressed_annotation)
from utils import (
    decompose_affine,
    write_image_orientation,
    correct_coordinate_transforms_rfc5,
    round_transform_values,
)


@dataclass
class AnnotationSet(AtlasAsset):
    """Brain structure annotation dataset manager.

    Attributes:
        template: Aligned template
        terminology: Brain structure terminology/hierarchy
        scales: Available resolution scales in micrometers per voxel
    """

    template: Template
    terminology: Terminology
    scales: tuple

    _asset_location: ClassVar[str] = "annotation-sets"
    schema_version: ClassVar[str] = "0.1.0"

    @property
    def manifest(self) -> dict:
        """Generate manifest dictionary for this annotation set."""
        return super().manifest | {
            "template": self.template.manifest,
            "terminology": self.terminology.manifest,
            "scales": list(self.scales),
        }

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "AnnotationSet":
        template = Template.from_manifest(manifest["template"], root=root)
        terminology = Terminology.from_manifest(manifest["terminology"], root=root)
        scales = tuple(manifest.get("scales", ()))
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            template=template,
            terminology=terminology,
            scales=scales,
        )

    def create(self, compressed_results, output_root, include_meshes=True):
        """Create annotation set from compressed annotation data at multiple scales.

        Args:
            compressed_results: Dictionary mapping scale to (compressed_data, affine) tuples
            output_root: Root directory where the annotation set will be created
        """
        output_dir = self.location(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Creating annotation set at {output_dir}")

        # Convert compressed annotations to OME-Zarr multiscale format
        logging.info("Converting compressed anatomical annotations to OME-zarr...")
        convert_compressed_annotations_to_zarr(compressed_results, output_dir, scales=self.scales)

        # Create precomputed file using highest resolution scale
        highest_res_scale = min(self.scales)  # Smallest number = highest resolution
        logging.info(f"Creating precomputed annotation file using highest resolution scale: {highest_res_scale}μm")

        high_res_data, high_res_affine = compressed_results[highest_res_scale]
        scale_vec, rotation_mat, translation_vec = decompose_affine(high_res_affine)  # Ensure affine is decomposed correctly
        # Round scale
        scale_vec = [round(1e4*dim)/1e4 for dim in scale_vec]
        logging.info(
            f"Decomposed affine for highest resolution: scale={scale_vec}, translation={translation_vec}, rotation=\n{rotation_mat}"
        )
        precomputed_output = output_dir / "annotations.precomputed"
        convert_compressed_annotations_to_precomputed(
            high_res_data,
            str(precomputed_output),
            scale=scale_vec,
        )

        # Write segment properties (without meshes) from terminology
        logging.info("Writing segment properties (without meshes) to precomputed...")
        write_segment_properties(
            precomputed_output,
            self.terminology.df,
        )

        if include_meshes:
            # Create mesh precomputed objects from annotation
            logging.info(f"Creating meshes from annotation to {precomputed_output}")
            create_mesh_from_compressed_annotation(high_res_data, self.scales, self.terminology.df, precomputed_output)

        # Compute voxel counts for all terms using highest resolution data
        logging.info("Computing voxel counts for all terms at highest resolution...")
        voxel_counts_df = self.count_voxels_for_all_terms(high_res_data, highest_res_scale)

        # Save voxel counts to a CSV file for reference
        voxel_counts_file = output_dir / "parcellation_volumes.csv"
        voxel_counts_df.to_csv(voxel_counts_file, index=False)
        logging.info(f"Saved voxel counts for {len(voxel_counts_df)} terms to {voxel_counts_file}")

        logging.info(f"Created annotation set at {output_dir}")

    def create_from_nifti(self, input_prefix, output_root, include_meshes=True):
        """Create annotation set from NIfTI source files with standardized naming convention.

        Args:
            input_prefix: Prefix for input NIfTI files (files named {input_prefix}_{scale}.nii.gz)
            output_root: Root directory where the annotation set will be created
        """
        output_dir = self.location(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Creating annotation set from NIfTI files at {output_dir}")

        # Copy annotation files with standardized compressed annotation naming convention
        for scale in self.scales:
            src_file = f"{input_prefix}_{scale}.nii.gz"
            dst_file = output_dir / f"annotations_compressed_{scale}.nii.gz"
            shutil.copy2(src_file, dst_file)
            logging.info(f"Copied {src_file} to {dst_file}")

        # Load the copied files and use the general create method
        compressed_results = load_compressed_annotations(output_dir, scales=self.scales)
        self.create(compressed_results, output_root, include_meshes=include_meshes)

    def create_from_mhd(self, mhd_paths, output_root, include_meshes=True):
        """Create annotation set from MHD (Meta Image) source files.

        Args:
            mhd_paths: Dictionary mapping scale to MHD file paths, or single MHD path for single scale
            output_root: Root directory where the annotation set will be created
        """

        output_dir = self.location(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Creating annotation set from MHD files at {output_dir}")

        # Handle single MHD file or multiple files
        if isinstance(mhd_paths, (str, Path)):
            # Single MHD file - assume it matches the single scale
            if len(self.scales) != 1:
                raise ValueError("Single MHD path provided but annotation set has multiple scales")
            mhd_paths = {self.scales[0]: mhd_paths}

        # Convert MHD files to NIfTI with proper naming and collect compressed results
        compressed_results = {}

        for scale in self.scales:
            if scale not in mhd_paths:
                logging.warning(f"No MHD file provided for scale {scale}, skipping")
                continue

            mhd_path = mhd_paths[scale]
            logging.info(f"Processing MHD file for scale {scale}: {mhd_path}")

            # Read MHD file using SimpleITK
            image = sitk.ReadImage(str(mhd_path))

            # Convert spacing from microns to millimeters (divide by 1000)
            spacing_microns = image.GetSpacing()
            spacing_mm = tuple(s / 1000.0 for s in spacing_microns)
            image.SetSpacing(spacing_mm)

            # Convert origin from microns to millimeters (divide by 1000)
            origin_microns = image.GetOrigin()
            origin_mm = tuple(o / 1000.0 for o in origin_microns)
            image.SetOrigin(origin_mm)

            # Save as temporary NIfTI file
            temp_nii = output_dir / f"annotations_compressed_{scale}.nii.gz"
            sitk.WriteImage(image, str(temp_nii))
            logging.info(f"Converted MHD to NIfTI: {temp_nii}")

            # Load the converted file to get compressed results
            img = nib.load(str(temp_nii))
            compressed_data = img.get_fdata().astype(np.int32)
            compressed_results[scale] = (compressed_data, img.affine)

            logging.info(f"Loaded compressed annotation for scale {scale} with shape {compressed_data.shape}")

        if not compressed_results:
            raise ValueError("No valid MHD files were processed")

        # Use the general create method with the compressed results
        self.create(compressed_results, output_root, include_meshes=include_meshes)

    def count_voxels_for_all_terms(self, compressed_annotation_data, spacing):
        """Count voxels for all terms in the terminology and return results as a DataFrame.

        Args:
            compressed_annotation_data: Pre-loaded compressed annotation data as numpy array
            spacing: N-dimensional spacing (voxel size) in micrometers, either a scalar or array-like

        Returns:
            pd.DataFrame: DataFrame with columns 'identifier', 'voxel_count', and 'volume_mm3'

        Raises:
            ValueError: If terminology doesn't have descendant_annotation_values column
        """
        if (
            "descendant_annotation_values"
            not in self.terminology.df.columns
        ):
            raise ValueError(
                "Terminology does not have 'descendant_annotation_values' column. "
                "Please ensure set_descendant_annotation_values() has been called."
            )

        # Convert spacing from micrometers to millimeters and compute voxel volume
        spacing_array = np.asarray(spacing) / 1000.0  # Convert μm to mm
        if spacing_array.ndim == 0:
            # Scalar spacing - assume isotropic
            voxel_volume = float(spacing_array) ** compressed_annotation_data.ndim
        else:
            # Array spacing - compute product
            voxel_volume = np.prod(spacing_array)

        logging.info(
            f"Counting voxels for all {len(self.terminology.df)} terms in terminology"
        )
        logging.info(
            f"Voxel spacing: {spacing_array} mm, voxel volume: {voxel_volume} mm³"
        )

        # Create a mapping from annotation value to list of term identifiers that should include it
        value_to_terms = {}
        term_to_identifier = {}

        for _, row in self.terminology.df.iterrows():
            term_identifier = row["identifier"]
            descendant_values = row["descendant_annotation_values"]
            term_to_identifier[term_identifier] = term_identifier

            if descendant_values:
                for value in descendant_values:
                    if value not in value_to_terms:
                        value_to_terms[value] = []
                    value_to_terms[value].append(term_identifier)

        # Initialize voxel counts for all terms
        voxel_counts = {term_id: 0 for term_id in term_to_identifier.keys()}

        # Get unique values in the annotation data (excluding background 0)
        unique_values, counts = np.unique(compressed_annotation_data, return_counts=True)

        logging.info(f"Found {len(unique_values)} unique annotation values, processing voxel counts...")

        # For each unique annotation value, add its count to all terms that include it
        for annotation_value, count in zip(unique_values, counts):
            if annotation_value == 0:  # Skip background
                continue

            if annotation_value in value_to_terms:
                for term_identifier in value_to_terms[annotation_value]:
                    voxel_counts[term_identifier] += count

        # Create results DataFrame
        results = []
        for term_identifier, voxel_count in voxel_counts.items():
            volume_mm3 = voxel_count * voxel_volume
            results.append(
                {
                    "identifier": term_identifier,
                    "voxel_count": voxel_count,
                    "volume_mm3": volume_mm3,
                }
            )

        df = pd.DataFrame(results)

        # Log summary statistics
        total_voxels = df["voxel_count"].sum()
        total_volume = df["volume_mm3"].sum()
        non_zero_counts = (df["voxel_count"] > 0).sum()
        logging.info(
            f"Voxel count summary: {total_voxels} total voxels, {total_volume:.2f} mm³ total volume across {non_zero_counts} terms with non-zero counts"
        )

        logging.info("Completed voxel counting for all terms")
        return df


def compute_term_mask(descendant_values, compressed_annotation_data):
    """Compute a boolean mask for voxels that match any of the given annotation values.

    Args:
        descendant_values: List of annotation values to match
        compressed_annotation_data: Compressed annotation data as numpy array

    Returns:
        numpy.ndarray: Boolean mask where True indicates voxels matching any of the values
    """
    if descendant_values is None or len(descendant_values) == 0:
        return np.zeros_like(compressed_annotation_data, dtype=bool)

    # Create a boolean mask for all descendant values
    mask = np.zeros_like(compressed_annotation_data, dtype=bool)
    for value in descendant_values:
        mask |= compressed_annotation_data == value

    return mask


def count_voxels_for_values(descendant_values, compressed_annotation_data):
    """Count voxels that match any of the given annotation values.

    Args:
        descendant_values: List of annotation values to count
        compressed_annotation_data: Compressed annotation data as numpy array

    Returns:
        int: Total number of voxels matching any of the values
    """
    mask = compute_term_mask(descendant_values, compressed_annotation_data)
    return int(np.sum(mask))


def uncompress_single_scale(
    compressed_data,
    affine,
    scale,
    terminology,
    zarr_group,
    zarr_dataset_name,
    zarr_chunks=(1, 128, 128, 128),
):
    """Decompress compressed annotation data into hierarchical annotations for a single scale."""
    # Check if terminology has descendant_annotation_values column
    if "descendant_annotation_values" not in terminology.df.columns:
        raise ValueError(
            "Terminology does not have 'descendant_annotation_values' column. "
            "Please ensure set_descendant_annotation_values() has been called."
        )

    # Get unique annotation values present in the data (excluding 0 which is background)
    data_annotation_values = set(np.unique(compressed_data))
    data_annotation_values.discard(0)  # Remove background

    logging.info(f"Found {len(data_annotation_values)} unique annotation values in data")

    # Process all identifiers in the terminology
    if "annotation_value" not in terminology.df.columns:
        raise ValueError(
            "Terminology does not have 'annotation_value' column. "
            "Please ensure annotation values are present in the terminology."
        )

    identifiers_with_data = []
    for _, row in terminology.df.iterrows():
        identifier = row["identifier"]
        annotation_value = row["annotation_value"]
        descendant_values = row["descendant_annotation_values"]

        # Check which descendant values are present in the data
        if descendant_values:
            present_descendants = [val for val in descendant_values if val in data_annotation_values]
        else:
            present_descendants = []
        # Include all identifiers for consistent dimensions (will be all zeros if no descendants present)
        identifiers_with_data.append((identifier, annotation_value, present_descendants))

    logging.info(f"Processing {len(identifiers_with_data)} identifiers from terminology")

    # Create uncompressed array: (identifiers, z, y, x)
    identifiers = [s[0] for s in identifiers_with_data]
    annotation_values = [s[1] for s in identifiers_with_data]
    uncompressed_shape = (len(identifiers),) + compressed_data.shape

    # Create affine matrix for 4D data
    new_affine = np.eye(4)
    new_affine[:3, :3] = affine[:3, :3]
    new_affine[:3, 3] = affine[:3, 3]

    logging.info(f"Creating zarr array and writing annotations in batches for scale {scale}")

    # Create zarr array
    zarr_array = zarr_group.create_dataset(
        zarr_dataset_name,
        shape=uncompressed_shape,
        chunks=zarr_chunks,
        dtype=compressed_data.dtype,
        compressors=(zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=1),),
    )

    # Process identifiers in batches to manage memory
    batch_size = 10  # Process 10 identifiers at a time
    for batch_start in range(0, len(identifiers_with_data), batch_size):
        batch_end = min(batch_start + batch_size, len(identifiers_with_data))
        batch_identifiers = identifiers_with_data[batch_start:batch_end]

        # Create batch array
        batch_annotations = np.zeros((len(batch_identifiers),) + compressed_data.shape, dtype=np.uint8)

        for i, (identifier, _, descendant_values) in enumerate(batch_identifiers):
            logging.info(
                f"Creating annotation for identifier {identifier} with {len(descendant_values)} descendant values"
            )

            # Use compute_term_mask to create the annotation
            if len(descendant_values) > 0:
                mask = compute_term_mask(descendant_values, compressed_data)
                batch_annotations[i] = mask.astype(np.uint8)

        # Write batch to zarr array
        start_idx = batch_start
        end_idx = batch_start + len(batch_identifiers)
        zarr_array[start_idx:end_idx] = batch_annotations

        # Clear batch from memory
        del batch_annotations

    return (zarr_array, new_affine, np.array(annotation_values))


def convert_compressed_annotations_to_zarr(compressed_results, output_dir, scales=(10, 25, 50, 100)):
    """Convert compressed annotation data to OME-Zarr multiscale format.

    Args:
        compressed_results (dict): Dictionary mapping scale to (compressed_data, affine) tuples
        output_dir (Path): Output directory where annotations_compressed.ome.zarr will be written
        scales (tuple): Scales to include in the multiscale dataset (default: (10, 25, 50, 100))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zarr_path = output_dir / "annotations_compressed.ome.zarr"

    logging.info("Converting compressed annotation data to OME-Zarr multiscale format")
    logging.info(f"Output Zarr path: {output_zarr_path}")

    arrays = []
    transforms = []
    axes = [
        {"name": "z", "type": "space", "unit": "millimeter"},
        {"name": "y", "type": "space", "unit": "millimeter"},
        {"name": "x", "type": "space", "unit": "millimeter"},
    ]

    for scale in scales:
        if scale in compressed_results:
            data, affine = compressed_results[scale]

            logging.info(f"Processing compressed scale {scale}: data shape {data.shape}, dtype {data.dtype}")
            arrays.append(data)

            # Extract transformation information from affine matrix
            scale_vec, rotation_mat, translation_vec = decompose_affine(affine)
            scale_vec = round_transform_values(scale_vec, decimals=6)
            translation_vec = round_transform_values(translation_vec, decimals=6)
            rotation_mat = round_transform_values(rotation_mat, decimals=8)

            # Write affine information to axes
            path_str = str(output_dir).lower()
            axes_orientation, original_orientation, ax_code = write_image_orientation(affine, axes, path_str)
            logging.info(f"Image axis: {ax_code}.\n Axis orientation is set to: {axes_orientation}")
            scale_transforms = []
            if scale_vec is not None:
                scale_transforms.append({"type": "scale", "scale": scale_vec.tolist()})
            if translation_vec is not None:
                scale_transforms.append({"type": "translation", "translation": translation_vec.tolist()})
            if rotation_mat is not None:
                scale_transforms.append({"type": "rotation", "rotation": rotation_mat.tolist()})
            transforms.append(scale_transforms)
        else:
            logging.warning(f"Scale {scale} not found in compressed results, skipping.")

    if arrays:
        group = zarr.open(str(output_zarr_path), mode="w")        
        labels_grp = group.require_group("labels")
        annotations_grp = labels_grp.require_group("annotations_compressed")

        logging.info("Writing OME-Zarr multiscale compressed annotations with chunk size (128, 128, 128)...")

        # Dictionary format for ome-zarr write_multiscale
        compressor_dict = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}

        write_multiscale(
            arrays,
            annotations_grp,
            axes=original_orientation,
            coordinate_transformations=transforms,
            chunks=(128, 128, 128),  # 3D chunks for compressed data
            compressor=compressor_dict,
        )

        correct_coordinate_transforms_rfc5(annotations_grp, axes_orientation)

        logging.info(f"OME-Zarr multiscale compressed annotations written to {output_zarr_path}")
    else:
        logging.error("No compressed annotation data found to convert")


def load_compressed_annotations(input_dir, scales=(10, 25, 50, 100)):
    """Load compressed annotation files from disk.

    Returns:
        dict: Dictionary mapping scale to (compressed_data, affine) tuples,
              where compressed_data is a numpy array and affine is the 4x4 transform matrix
    """
    results = {}

    for scale in scales:
        src_fname = f"annotations_compressed_{scale}.nii.gz"
        src = input_dir / src_fname

        if not src.exists():
            logging.warning(f"Source file {src} does not exist, skipping.")
            continue

        logging.info(f"Loading compressed annotation file: {src}")
        img = nib.load(str(src))
        compressed_data = img.get_fdata().astype(np.int32)

        results[scale] = (compressed_data, img.affine)
        logging.info(f"Loaded compressed annotation for scale {scale} with shape {compressed_data.shape}")

    return results


def uncompress_annotations_to_zarr(input_dir, terminology, output_dir, scales=(10, 25, 50, 100)):
    """Create hierarchical binary annotations from compressed annotations with memory optimization."""
    # Descendants are already precomputed in the terminology object

    output_dir.mkdir(parents=True, exist_ok=True)
    output_zarr_path = output_dir / "annotations.ome.zarr"

    logging.info("Creating memory-optimized OME-Zarr multiscale format with consistent dimensions")
    logging.info(f"Output Zarr path: {output_zarr_path}")

    # Remove existing zarr file if it exists
    if output_zarr_path.exists():
        logging.info(f"Removing existing zarr file: {output_zarr_path}")
        shutil.rmtree(output_zarr_path)

    # Create zarr group
    group = zarr.open(str(output_zarr_path), mode="w")
    labels_grp = group.require_group("labels")
    annotations_grp = labels_grp.require_group("annotations")

    # Process each scale and create zarr arrays directly
    zarr_arrays = []
    transforms = []
    annotation_values = None  # Will be set from the first scale processed

    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    axes_orientation = None
    original_orientation = None

    scale_index = 0
    for scale in scales:
        src_fname = f"annotations_compressed_{scale}.nii.gz"
        src = input_dir / src_fname

        if not src.exists():
            logging.warning(f"Source file {src} does not exist, skipping.")
            continue

        logging.info(f"Processing scale {scale}: {src}")
        img = nib.load(str(src))
        compressed_data = img.get_fdata().astype(np.int32)

        # Use uncompress_single_scale with zarr group to create array directly
        result = uncompress_single_scale(
            compressed_data,
            img.affine,
            scale,
            terminology,
            zarr_group=annotations_grp,
            zarr_dataset_name=f"{scale_index}",
            zarr_chunks=(1, 128, 128, 128),
        )

        if result is None:
            continue

        zarr_array, new_affine, values = result
        zarr_arrays.append(zarr_array)

        # Set annotation_values from the first scale processed (all scales have same values)
        if annotation_values is None:
            annotation_values = values

        # Extract transformation information from affine matrix
        scale_vec, rotation_mat, translation_vec = decompose_affine(new_affine)
        scale_vec = round_transform_values(scale_vec, decimals=6)
        translation_vec = round_transform_values(translation_vec, decimals=6)
        rotation_mat = round_transform_values(rotation_mat, decimals=8)

        # Update axes orientation metadata once using spatial axes
        if axes_orientation is None or original_orientation is None:
            path_str = str(output_dir).lower()
            spatial_axes = [dict(axes[1]), dict(axes[2]), dict(axes[3])]
            spatial_axes_orientation, spatial_original_orientation, _ = write_image_orientation(
                new_affine,
                spatial_axes,
                path_str,
            )
            axes_orientation = [axes[0]] + spatial_axes_orientation
            original_orientation = [axes[0]] + spatial_original_orientation

        scale_transforms = []

        # For 4D data, prepend identity for channel dimension
        if scale_vec is not None:
            full_scale = [1.0] + scale_vec.tolist()
            scale_transforms.append({"type": "scale", "scale": full_scale})

        if translation_vec is not None:
            full_translation = [0.0] + translation_vec.tolist()
            scale_transforms.append({"type": "translation", "translation": full_translation})

        if rotation_mat is not None:
            rotation_4d = np.eye(4)
            rotation_4d[1:, 1:] = rotation_mat
            scale_transforms.append({"type": "rotation", "rotation": rotation_4d.tolist()})

        transforms.append(scale_transforms)

        scale_index += 1

    if not zarr_arrays:
        logging.error("No valid scales found")
        return

    # Set up OME-Zarr metadata

    # Write metadata
    write_multiscales_metadata(
        annotations_grp,
        datasets=[
            {
                "path": str(i),
                "coordinateTransformations": transforms[i],
            }
            for i in range(len(zarr_arrays))
        ],
        axes=original_orientation if original_orientation is not None else axes,
    )

    if axes_orientation is not None:
        correct_coordinate_transforms_rfc5(annotations_grp, axes_orientation)

    # Store annotation_values as a separate array in the zarr group
    if annotation_values is not None:
        annotations_grp.create_dataset(
            "annotation_values",
            shape=annotation_values.shape,
            data=annotation_values,
            dtype=annotation_values.dtype,
            compressors=(zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=1),),
        )
        logging.info(f"Stored {len(annotation_values)} annotation values")

    logging.info(f"Memory-optimized OME-Zarr multiscale annotations written to {output_zarr_path}")
