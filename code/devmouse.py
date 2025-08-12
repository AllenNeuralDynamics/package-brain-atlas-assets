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

from atlas_builder import (
    AnatomicalAnnotationSet,
    AnatomicalSpace,
    AnatomicalTemplate,
    ParcellationAtlas,
    ParcellationTerminology,
)
import datetime
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema_models.data_name_patterns import build_data_name
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema.components.identifiers import Person


DEVMOUSE_ONTOLOGY_DESCRIPTION = "The Allen Developing Mouse Brain Atlas ontology, authored by Professor Luis Puelles, M.D., Ph.D., organizes mouse brain structures developmentally from the earliest embryonic stage to adulthood using a topological rather than fixed-coordinate approach, enabling applicability to both developing and mature forms. Beginning at Level 00 with the unpatterned neural plate, it progresses through 13 hierarchical levels defined by permanent early boundaries, internal landmarks, and gene expression patterns. Early levels (01–05) capture broad divisions—forebrain, midbrain, hindbrain, spinal cord—followed by neuromeric and dorsoventral partitioning. Intermediate levels (06–08) refine subdivisions, especially in the telencephalon, while Levels 09–10 address radial layering of the neural wall. The final stages (11–13) classify nuclei and subnuclei, largely following The Mouse Brain in Stereotaxic Coordinates by Franklin and Paxinos (2008), with refinements from the ontology’s planar framework. This developmental, topology-based classification facilitates consistent mapping across stages and species, linking embryonic and adult brain data."
DEVMOUSE_TEMPLATE_DESCRIPTION = "For this developmental atlas (Age: {Age (days)}, Theiler stage: {Theiler stage}, Gender: {Gender}), a reference set of tissue preparations was generated in the {Plane} plane with a histological stain ({Stain}) to aid identification of anatomical structures for atlas drawing. The specimen used was a {Specimen}, sectioned at {Section width} thickness. Annotation was performed on the {Annotated hemisphere} hemisphere, with {# Annotated images} images annotated. Embryonic (E) specimen age is provided relative to days after conception, with birth expected at approximately 19 days post-conception. Postnatal (P) specimen age is given relative to birth (P0). Theiler stages were determined on the basis of external features identified during dissection and embedding (Theiler, 1989). HP Yellow, a nuclear stain, was used for whole embryo reference sets to allow visualization of all tissues and cells; this stain is also used as a counterstain for the ISH in the Allen Developing Mouse Brain Atlas. Nissl stains were used for all dissected brains to provide additional morphological information of maturing neurons. To make a coherent 3D volume, section images were coregistered to each other."
DEVMOUSE_TEMPLATE_DATA = [
    {"Age (days)": "E11.5", "Theiler stage": "TS19", "Gender": "N.D.", "Plane": "sagittal",
     "Stain": "HP Yellow", "Specimen": "Whole embryo", "Section width": "20 µm",
     "Annotated hemisphere": "Right", "# Annotated images": 28},
    {"Age (days)": "E13.5", "Theiler stage": "TS21", "Gender": "N.D.", "Plane": "sagittal",
     "Stain": "HP Yellow", "Specimen": "Whole embryo", "Section width": "20 µm",
     "Annotated hemisphere": "Right", "# Annotated images": 15},
    {"Age (days)": "E15.5", "Theiler stage": "TS24", "Gender": "male", "Plane": "sagittal",
     "Stain": "HP Yellow", "Specimen": "Whole embryo", "Section width": "20 µm",
     "Annotated hemisphere": "Right", "# Annotated images": 16},
    {"Age (days)": "E18.5", "Theiler stage": "TS26", "Gender": "male", "Plane": "sagittal",
     "Stain": "Nissl (cresyl violet)", "Specimen": "Dissected brain", "Section width": "20 µm",
     "Annotated hemisphere": "Left", "# Annotated images": 19},
    {"Age (days)": "P4", "Theiler stage": "-", "Gender": "male", "Plane": "sagittal",
     "Stain": "Nissl (cresyl violet)", "Specimen": "Dissected brain", "Section width": "20 µm",
     "Annotated hemisphere": "Left", "# Annotated images": 23},
    {"Age (days)": "P14", "Theiler stage": "-", "Gender": "male", "Plane": "sagittal",
     "Stain": "Nissl (thionin)", "Specimen": "Dissected brain", "Section width": "25 µm",
     "Annotated hemisphere": "Left", "# Annotated images": 39},
    {"Age (days)": "P56", "Theiler stage": "-", "Gender": "male", "Plane": "sagittal",
     "Stain": "Nissl (thionin)", "Specimen": "Dissected brain", "Section width": "25 µm",
     "Annotated hemisphere": "Left", "# Annotated images": 21},
]

# Creation time constants
DEVMOUSE_ONTOLOGY_CREATION_TIME = datetime.datetime(2012, 1, 1, tzinfo=datetime.timezone.utc)
DEVMOUSE_TEMPLATE_CREATION_TIME = datetime.datetime(2012, 1, 1, tzinfo=datetime.timezone.utc)


def _write_devmouse_ontology_data_description(output_dir: Path):
    """Write data_description.json for the developmental mouse ontology."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name("allen-dev-mouse-terminology", DEVMOUSE_ONTOLOGY_CREATION_TIME),
        data_summary=DEVMOUSE_ONTOLOGY_DESCRIPTION.strip(),
        subject_id="developing-mouse",
        modalities=[Modality.BRIGHTFIELD],  # Source modalities (histological stains)
        data_level="derived",
        creation_time=DEVMOUSE_ONTOLOGY_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Lydia Ng", registry_identifier="0000-0002-7499-3514")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Developing Mouse Brain Atlas",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote ontology data_description.json to {output_dir}")


def _lookup_template_row(age_token: str):
    """Return row dict from DEVMOUSE_TEMPLATE_DATA matching age token like 'E11pt5'."""
    age_display = age_token.replace("pt", ".")  # Convert E11pt5 -> E11.5
    for row in DEVMOUSE_TEMPLATE_DATA:
        if row.get("Age (days)") == age_display:
            return row
    return None


def _write_devmouse_template_data_description(output_dir: Path, age_token: str):
    """Write data_description.json for a developmental mouse template for a specific age."""
    row = _lookup_template_row(age_token)
    if row is None:
        logging.warning(f"No template metadata row found for age {age_token}; skipping data description")
        return
    summary = DEVMOUSE_TEMPLATE_DESCRIPTION.format(**row)
    subject_id = f"dev-mouse-{age_token.lower()}"
    dd = DataDescription(
        name=build_data_name(f"allen-dev-mouse-{age_token.lower()}-template", DEVMOUSE_TEMPLATE_CREATION_TIME),
        data_summary=summary.strip(),
        subject_id=subject_id,
        modalities=[Modality.BRIGHTFIELD],
        data_level="derived",
        creation_time=DEVMOUSE_TEMPLATE_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Lydia Ng", registry_identifier="0000-0002-7499-3514")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Developing Mouse Brain Atlas",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote template data_description.json for age {age_token} to {output_dir}")


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

    # Write ontology data description
    _write_devmouse_ontology_data_description(terminology.location(output_dir))

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

    # Write template data description
    _write_devmouse_template_data_description(template.location(results_dir), age)

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
