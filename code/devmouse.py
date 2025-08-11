#!/usr/bin/env python3
"""
Package DevMouse atlas assets using AssetLibrary.

This script packages the developmental mouse atlas assets from the Allen Institute
into the AssetLibrary format, handling MHD files and creating proper anatomical
templates and annotation sets.
"""

import logging
import os
import shutil
import traceback
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from atlas_assets import (AnatomicalAnnotationSet, AnatomicalSpace,
                          AnatomicalTemplate, ParcellationAtlas,
                          ParcellationTerminology)


def read_mhd_metadata(mhd_path):
    """
    Read metadata from an MHD file using SimpleITK.

    Parameters
    ----------
    mhd_path : str
        Path to the MHD file

    Returns
    -------
    tuple
        (spacing, origin, direction)
    """
    # Read the image using SimpleITK (metadata only)
    image = sitk.ReadImage(str(mhd_path))

    # Get metadata
    spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()  # (x, y, z)
    direction = image.GetDirection()  # 9-element tuple for 3D

    return spacing, origin, direction


def convert_mhd_to_nifti(mhd_path, output_path):
    """
    Convert an MHD file to NIfTI format.

    Converts units from microns (MHD) to millimeters (NIfTI standard).

    Parameters
    ----------
    mhd_path : str
        Path to the input MHD file
    output_path : str
        Path for the output NIfTI file
    """
    # Read the MHD file
    image = sitk.ReadImage(str(mhd_path))

    # Convert spacing from microns to millimeters (divide by 1000)
    spacing_microns = image.GetSpacing()
    spacing_mm = tuple(s / 1000.0 for s in spacing_microns)
    image.SetSpacing(spacing_mm)

    # Convert origin from microns to millimeters (divide by 1000)
    origin_microns = image.GetOrigin()
    origin_mm = tuple(o / 1000.0 for o in origin_microns)
    image.SetOrigin(origin_mm)

    # Write as NIfTI
    sitk.WriteImage(image, str(output_path))
    print(f"Converted {mhd_path} to {output_path} (units converted from microns to mm)")


def create_devmouse_terminology(output_dir, library):
    """
    Create a ParcellationTerminology from the devmouse structures CSV.

    Parameters
    ----------
    output_dir : Path
        Output directory for the terminology
    library : AssetLibrary
        Asset library to add the terminology to
    """
    structures_path = Path(
        "/root/capsule/data/devmouse-atlas-assets/devmouse_structures.csv"
    )

    df = pd.read_csv(structures_path)

    # Create DataFrame with required columns for ParcellationTerminology
    structures_df = pd.DataFrame(
        {
            "identifier": df["id"].map(lambda x: f"DMBA:{int(x)}"),
            "annotation_value": df["id"].astype(int),
            "parent_identifier": df["parent_structure_id"].map(
                lambda x: f"DMBA:{int(x)}" if not pd.isna(x) else ""
            ),
            "name": df["name"],
            "abbreviation": df["acronym"],
            "color_hex_triplet": df["color_hex_triplet"].map(lambda x: f"#{x}"),
        }
    )

    # Create the terminology
    terminology = ParcellationTerminology(
        name="allen-dev-mouse-terminology", version="2012", df=structures_df
    )

    # Descendant annotation values require lookup since identifiers are prefixed
    id_to_ann = dict(
        zip(terminology.df["identifier"], terminology.df["annotation_value"])
    )
    terminology.set_descendant_annotation_values(
        lambda row: [id_to_ann[i] for i in row["descendants"] if i in id_to_ann]
    )

    # Save and add to library
    parcellation_legacy_dir = terminology.location(output_dir) / "legacy_files"
    parcellation_legacy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(structures_path, parcellation_legacy_dir / structures_path.name)

    terminology.write_terminology(output_dir)
    terminology.create_manifest(output_dir)
    library.add(terminology)

    return terminology


def package_age_group(age, base_dir, results_dir, asset_library, terminology):
    """
    Package atlas assets for a specific age group.

    Parameters
    ----------
    age : str
        Age identifier (e.g., 'E11pt5', 'P14')
    base_dir : str
        Base directory containing the devmouse assets
    results_dir : Path
        Results directory for output
    asset_library : AssetLibrary
        The asset library to add assets to
    terminology : ParcellationTerminology
        The parcellation terminology
    """
    print(f"\nProcessing age group: {age}")

    # Skip P56_Mouse files as requested
    if age.startswith("P56_Mouse"):
        print(f"Skipping {age} as requested")
        return

    # Define paths
    template_dir = os.path.join(base_dir, f"{age}_atlasVolume")
    annotation_dir = os.path.join(base_dir, f"{age}_DevMouse2012_annotation")

    # Check if directories exist
    if not os.path.exists(template_dir):
        print(f"Template directory not found: {template_dir}")
        return

    if not os.path.exists(annotation_dir):
        print(f"Annotation directory not found: {annotation_dir}")
        return

    # Find MHD files
    template_mhd = None
    annotation_mhd = None

    for root, dirs, files in os.walk(template_dir):
        for file in files:
            if file.endswith(".mhd"):
                template_mhd = os.path.join(root, file)
                break
        if template_mhd:
            break

    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith(".mhd"):
                annotation_mhd = os.path.join(root, file)
                break
        if annotation_mhd:
            break

    if not template_mhd:
        print(f"No MHD file found in {template_dir}")
        return

    if not annotation_mhd:
        print(f"No MHD file found in {annotation_dir}")
        return

    print(f"  Template: {template_mhd}")
    print(f"  Annotation: {annotation_mhd}")

    # Get voxel spacing from MHD file to determine appropriate scale
    spacing, _, _ = read_mhd_metadata(template_mhd)
    # Spacing is already in microns, so just round to nearest integer
    scale = int(round(max(spacing)))

    # Create anatomical template
    template_name = f"allen-dev-mouse-{age.lower()}-nissl-template"
    template = AnatomicalTemplate(name=template_name, version="2012", scales=(scale,))

    # Create template directory and convert MHD to NIfTI with correct naming
    template_dir = template.location(results_dir)
    template_dir.mkdir(parents=True, exist_ok=True)

    template_nii = template_dir / f"anatomical_template_{scale}.nii.gz"
    convert_mhd_to_nifti(template_mhd, template_nii)

    # Convert NIfTI to OME-Zarr and create manifest
    template.convert_nifti_to_omezarr_multiscale(results_dir)
    template.create_manifest(results_dir)
    asset_library.add(template)
    print(f"  Added template: {template_name}")

    # Create annotation set
    annotation_name = f"allen-dev-mouse-{age.lower()}-annotation"
    annotation_set = AnatomicalAnnotationSet(
        name=annotation_name,
        anatomical_template=template,
        parcellation_terminology=terminology,
        version="2012",
        scales=(scale,),
    )

    # Create annotation set using the MHD file directly
    annotation_set.create_from_mhd(annotation_mhd, results_dir)

    annotation_set.create_manifest(results_dir)
    asset_library.add(annotation_set)
    print(f"  Added annotation set: {annotation_name}")

    # Create anatomical space for this developmental stage
    space_name = f"allen-dev-mouse-{age.lower()}-space"
    anatomical_space = AnatomicalSpace(
        name=space_name, version="2012", anatomical_template=template
    )
    anatomical_space.create_manifest(results_dir)
    asset_library.add(anatomical_space)
    print(f"  Created anatomical space: {space_name}")

    # Create parcellation atlas
    atlas_name = f"allen-dev-mouse-{age.lower()}-atlas"
    atlas = ParcellationAtlas(
        name=atlas_name,
        version="2012",
        anatomical_space=anatomical_space,
        anatomical_annotation_set=annotation_set,
        parcellation_terminology=terminology,
    )
    atlas.create_manifest(results_dir)
    asset_library.add(atlas)
    print(f"  Created parcellation atlas: {atlas_name}")


def package_devmouse(base_dir, results_dir, library):
    """Package all devmouse assets using the provided library and directories."""
    logging.info("Starting DevMouse atlas packaging...")

    # Create terminology
    logging.info("Creating parcellation terminology...")
    terminology = create_devmouse_terminology(results_dir, library)

    # Define age groups to process (excluding gridAnnotation and P56_Mouse files)
    age_groups = ["E11pt5", "E13pt5", "E15pt5", "E16pt5", "E18pt5", "P4", "P14", "P28"]

    # Process each age group (creates templates, annotations, spaces, and atlases)
    for age in age_groups:
        try:
            package_age_group(age, str(base_dir), results_dir, library, terminology)
        except Exception as e:
            logging.error(f"Error processing DevMouse age {age}: {e}")
            traceback.print_exc()
            continue

    logging.info("DevMouse atlas packaging complete!")

    # Log summary
    logging.info("DevMouse Summary:")
    logging.info(
        f"  Templates: {len([t for t in library.anatomical_templates if 'dev-mouse' in t.name])}"
    )
    logging.info(
        f"  Annotation sets: {len([a for a in library.anatomical_annotation_sets if 'dev-mouse' in a.name])}"
    )
    logging.info(
        f"  Anatomical spaces: {len([s for s in library.anatomical_spaces if 'dev-mouse' in s.name])}"
    )
    logging.info(
        f"  Parcellation atlases: {len([p for p in library.parcellation_atlases if 'dev-mouse' in p.name])}"
    )
