#!/usr/bin/env python3
"""
Package DevMouse atlas assets using AssetLibrary.

This script packages the developmental mouse atlas assets from the Allen Institute
and Kim Lab into the AssetLibrary format from input anatomical
templates and annotation set volumes.
"""

import logging
import os
import shutil
import traceback
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from atlas_builder import (
    AnnotationSet,
    CoordinateSpace,
    Template,
    Atlas,
    Terminology,
)
import datetime
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema_models.data_name_patterns import build_data_name
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema.components.identifiers import Person


DEVMOUSE_ONTOLOGY_DESCRIPTION = "The Allen Developing Mouse Brain Atlas ontology, authored by Professor Luis Puelles, M.D., Ph.D., organizes mouse brain structures developmentally from the earliest embryonic stage to adulthood using a topological rather than fixed-coordinate approach, enabling applicability to both developing and mature forms. Beginning at Level 00 with the unpatterned neural plate, it progresses through 13 hierarchical levels defined by permanent early boundaries, internal landmarks, and gene expression patterns. Early levels (01–05) capture broad divisions—forebrain, midbrain, hindbrain, spinal cord—followed by neuromeric and dorsoventral partitioning. Intermediate levels (06–08) refine subdivisions, especially in the telencephalon, while Levels 09–10 address radial layering of the neural wall. The final stages (11–13) classify nuclei and subnuclei, largely following The Mouse Brain in Stereotaxic Coordinates by Franklin and Paxinos (2008), with refinements from the ontology’s planar framework. This developmental, topology-based classification facilitates consistent mapping across stages and species, linking embryonic and adult brain data."
DEVMOUSE_TEMPLATE_DESCRIPTION = "For this developmental atlas (Age: {Age (days)}, a reference lightsheet template was created from {sample number} individuals ({number female} female) using {Specimen} fixed samples. Individual samples were co-registered using a landmark-assisted multimodal registration method, registered to an MRI template space, and resampled to 20 µm isotropic voxel resolution. Annotation was performed manually using terminology from the Allen Developing Mouse Brain Atlas ontology. Embryonic (E) specimen age is provided relative to days after conception, with birth expected at approximately 19 days post-conception. Postnatal (P) specimen age is given relative to birth (P0). A full description of atlas creation is published in Kronman et al. (2024), https://doi.org/10.1038/s41467-024-53254-w"
DEVMOUSE_TEMPLATE_DATA = [
    {
        "Age (days)": "E11.5",
        "Specimen": "Whole embryo",
        "sample number": "10",
        "number female": "5",
    },
    {
        "Age (days)": "E13.5",
        "Specimen": "Whole embryo",
        "sample number": "10",
        "number female": "5",
    },
    {
        "Age (days)": "E15.5",
        "Specimen": "Whole embryo",
        "sample number": "9",
        "number female": "4",
    },   
    {
        "Age (days)": "E18.5",
        "Specimen": "Dissected brain",
        "sample number": "9",
        "number female": "4",
    },
    {
        "Age (days)": "P4",
        "Specimen": "Dissected brain",
        "sample number": "7",
        "number female": "4",
    },
    {
        "Age (days)": "P14",
        "Specimen": "Dissected brain",
        "sample number": "10",
        "number female": "5",
    },
    {
        "Age (days)": "P56",
        "Specimen": "Dissected brain",
        "sample number": "6",
        "number female": "3",
    },
]

# Creation time constants
DEVMOUSE_ONTOLOGY_CREATION_TIME = datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)
DEVMOUSE_TEMPLATE_CREATION_TIME = datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)


def _write_devmouse_ontology_data_description(output_dir: Path):
    """Write data_description.json for the developmental mouse ontology."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name("allen-dev-mouse-terminology", DEVMOUSE_ONTOLOGY_CREATION_TIME),
        data_summary=DEVMOUSE_ONTOLOGY_DESCRIPTION.strip(),
        subject_id="developing-mouse",
        modalities=[Modality.SPIM],  # Source modalities (histological stains)
        data_level="derived",
        creation_time=DEVMOUSE_ONTOLOGY_CREATION_TIME,
        institution=Organization.AIBS, # Need to add new organization to aind-data-schema-models
        investigators=[Person(name="Yongsoo Kim", registry_identifier="0000-0002-2995-2131")],
        funding_source=[Funding(funder=Organization.NINDS)],
        project_name="Allen Developing Mouse Brain Atlas",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote ontology data_description.json to {output_dir}")


def _lookup_template_row(age_token: str):
    """Return row dict from DEVMOUSE_TEMPLATE_DATA matching age token like 'E11pt5'."""
    age_display = age_token.replace("p", ".")  # Convert E11pt5 -> E11.5
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
        name=build_data_name(
            f"allen-dev-mouse-{age_token.lower()}-template",
            DEVMOUSE_TEMPLATE_CREATION_TIME,
        ),
        data_summary=summary.strip(),
        subject_id=subject_id,
        modalities=[Modality.SPIM],
        data_level="derived",
        creation_time=DEVMOUSE_TEMPLATE_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Yongsoo Kim", registry_identifier="0000-0002-2995-2131")],
        funding_source=[Funding(funder=Organization.NINDS)],
        project_name="Allen Developing Mouse Brain Atlas",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote template data_description.json for age {age_token} to {output_dir}")


def create_devmouse_terminology(output_dir, library):
    """
    Create a Terminology from the devmouse structures CSV.

    Parameters
    ----------
    output_dir : Path
        Output directory for the terminology
    library : AssetLibrary
        Asset library to add the terminology to
    """
    structures_path = Path("/root/capsule/data/devmouse-lsfm/terminology/2024/allen-dev-mouse-terminology-v004.csv")

    df = pd.read_csv(structures_path)

    # Create DataFrame with required columns for Terminology
    structures_df = pd.DataFrame(
        {
            "identifier": df["identifier"],
            "annotation_value": df["annotation_value"].astype(int),
            "parent_identifier": df["parent_identifier"],
            "name": df["name"],
            "abbreviation": df["abbreviation"],
            "color_hex_triplet": df["color_hex_triplet"],
        }
    )

    # Create the terminology
    terminology = Terminology(
        name="allen-dev-mouse-terminology", version="2024", df=structures_df
    )

    # Descendant annotation values require lookup since identifiers are prefixed
    id_to_ann = dict(zip(terminology.df["identifier"], terminology.df["annotation_value"]))
    terminology.set_descendant_annotation_values(
        lambda row: [id_to_ann[i] for i in row["descendant_identifiers"] if i in id_to_ann]
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


def package_age_group(age: str, base_dir: Path, results_dir: Path, asset_library, terminology):
    """
    Package atlas assets for a specific age group.

    Parameters
    ----------
    age : str
        Age identifier (e.g., 'E11p5', 'P14')
    base_dir : str
        Base directory containing the devmouse assets
    results_dir : Path
        Results directory for output
    asset_library : AssetLibrary
        The asset library to add assets to
    terminology : Terminology
        The parcellation terminology
    """
    print(f"\nProcessing age group: {age}")

    # Skip P56_Mouse files as requested
    if age.startswith("P56"):
        print(f"Skipping {age} as requested")
        return

    # Define paths
    template_dir = base_dir / f"templates/allen-dev-mouse-{age}-lsfm-template/2024"
    annotation_dir = base_dir / f"annotation-sets/allen-dev-mouse-{age}-annotation/2024"

    # Check if directories exist
    if not os.path.exists(template_dir):
        print(f"Template directory not found: {template_dir}")
        return

    if not os.path.exists(annotation_dir):
        print(f"Annotation directory not found: {annotation_dir}")
        return


    # Create anatomical template
    template_name = f"allen-dev-mouse-{age.lower()}-spim-template"
    template = Template(name=template_name, version="2024", scales=(20,))
    template.create(input_prefix = template_dir / "template", output_root = results_dir)
    asset_library.add(template)
    print(f"  Added template: {template_name}")

    # Write template data description
    _write_devmouse_template_data_description(template.location(results_dir), age)

    # Create annotation set
    annotation_name = f"allen-dev-mouse-{age.lower()}-spim-annotation"
    annotation_set = AnnotationSet(
        name=annotation_name,
        template=template,
        terminology=terminology,
        version="2024",
        scales=(20,),
    )

    annotation_set.create_from_nifti(
            input_prefix=annotation_dir / "annotation",
            output_root=results_dir,
            include_meshes=True
        )
    annotation_set.create_manifest(results_dir)
    asset_library.add(annotation_set)
    print(f"  Added annotation set: {annotation_name}")

    # Create coordinate space for this developmental stage
    space_name = f"allen-dev-mouse-{age.lower()}-spim-space"
    coordinate_space = CoordinateSpace(
        name=space_name, version="2024", template=template
    )
    coordinate_space.create_manifest(results_dir)
    asset_library.add(coordinate_space)
    print(f"  Created coordinate space: {space_name}")

    # Create parcellation atlas
    atlas_name = f"allen-dev-mouse-{age.lower()}-spim-atlas"
    atlas = Atlas(
        name=atlas_name,
        version="2024",
        coordinate_space=coordinate_space,
        annotation_set=annotation_set,
        terminology=terminology,
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
    age_groups = ["e11pt5", "e13pt5", "e15pt5",  "e18pt5", "p4", "p14"]

    # Process each age group (creates templates, annotations, spaces, and atlases)
    for age in age_groups:
        try:
            package_age_group(age, base_dir, results_dir, library, terminology)
        except Exception as e:
            logging.error(f"Error processing DevMouse age {age}: {e}")
            traceback.print_exc()
            continue

    logging.info("DevMouse atlas packaging complete!")

    # Log summary
    logging.info("DevMouse Summary:")
    logging.info(
        f"  Templates: {len([t for t in library.templates if 'dev-mouse' in t.name])}"
    )
    logging.info(
        f"  Annotation sets: {len([a for a in library.annotation_sets if 'dev-mouse' in a.name])}"
    )
    logging.info(
        f"  Coordinate spaces: {len([s for s in library.coordinate_spaces if 'dev-mouse' in s.name])}"
    )
    logging.info(
        f"  Atlases: {len([p for p in library.atlases if 'dev-mouse' in p.name])}"
    )
