"""SmartSPIM brain template packaging and coordinate transformation functions."""

import datetime
import json
import logging
import shutil

from aind_data_schema.components.identifiers import Code, DataAsset, Person
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema.core.processing import DataProcess, Processing
from aind_data_schema_models.data_name_patterns import build_data_name
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization

from allen_atlas_assets import (AnatomicalAnnotationSet, AnatomicalTemplate,
                                CoordinateTransformation)


def create_smartspim_annotation_set(input_dir, results_dir, library):
    """Create SmartSPIM anatomical annotation set with CCF labels mapped to SmartSPIM space.

    Args:
        input_dir (Path): Path to input directory containing ccf_annotation_to_template_moved_25.nii.gz
        results_dir (Path): Path to results directory for output
        library (AssetLibrary): Asset library to get template and terminology from
    """
    # Get required assets from library
    template = library.get_anatomical_template("allen-adult-mouse-spim-lca-template", "2024-05")
    terminology = library.get_parcellation_terminology(
        "allen-adult-mouse-terminology", "2017"
    )

    annotation_set = AnatomicalAnnotationSet(
        name="allen-adult-mouse-spim-lca-annotation",
        anatomical_template=template,
        parcellation_terminology=terminology,
        version="2024-05",
        scales=(25,),
    )
    annotation_set.create_from_nifti(
        input_prefix=input_dir / "ccf_annotation_to_template_moved",
        output_root=results_dir,
    )
    annotation_set.create_manifest(results_dir)
    library.add(annotation_set)

    logging.info(
        f"Created SmartSPIM annotation set at {template.location(results_dir)}"
    )


def create_smartspim_coordinate_transformations(input_dir, results_dir, library):
    """Create coordinate transformation assets for SmartSPIM to CCF registration."""
    # Get required assets from library
    template = library.get_anatomical_template("allen-adult-mouse-spim-lca-template", "2024-05")
    ccf_template = library.get_anatomical_template("allen-adult-mouse-2p-template", "2015")

    # Create coordinate transform from SmartSPIM template to CCF 3.1
    cs = CoordinateTransformation.init(
        input_template=template,
        output_template=ccf_template,
        version="2024-05",
    )
    transform_dir = cs.location(results_dir)
    transform_dir.mkdir(parents=True, exist_ok=True)
    copy_transform_files(input_dir, transform_dir)

    cs.create_manifest(results_dir)
    library.add(cs)

    logging.info(f"Created SmartSPIM coordinate transforms at {transform_dir}")


def copy_transform_files(input_dir, output_dir):
    """Copy ANTs transformation files with standardized naming."""
    # Look for files with "syn" in their name (ANTs transformation files)
    transform_pattern = "*syn*"
    transform_files = list(input_dir.glob(transform_pattern))

    if not transform_files:
        logging.warning(
            f"No transform files matching '{transform_pattern}' found in {input_dir}"
        )
        return

    for src in transform_files:
        # Remove the "spim_template_to_ccf_syn_" prefix if present for standardized naming
        new_name = src.name.replace("spim_template_to_ccf_syn_", "")
        dst = output_dir / new_name
        if not dst.exists():
            shutil.copy2(src, dst)
            logging.info(f"Copied transform file {src} to {dst} (renamed)")
        else:
            logging.info(f"Transform file {dst} already exists, skipping copy.")

    logging.info(f"Copied {len(transform_files)} transform files to {output_dir}")


def copy_data_description(input_dir, output_dir, name):
    """Copy and validate data description metadata for SmartSPIM template."""
    input_json_path = input_dir / "data_description.json"
    output_json_path = output_dir / "data_description.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json_path, "r") as f:
        data = json.load(f)

    t = datetime.datetime.fromisoformat(data["creation_time"])

    # Load and validate the data with the DataDescription model
    data_description = DataDescription(
        name=build_data_name(name.replace(".", "-"), t),
        data_summary="A computational average of 10 P56 mouse brains imaged on the LifeCanvas SmartSPIM (lightsheet) microscope at ~2um isotropic resolution after clearing using the LifeCanvas Active clearing protocol. Left-right flipped copies of all input images were included in the averaging process to symmetrize the resulting anatomical template. The template was computed at 25um resolution natively, then resampled to 10um, 50um, and 100um resolutions.",
        subject_id=data["subject_id"],
        modalities=[Modality.SPIM],
        institution=Organization.AIND,
        data_level="derived",
        creation_time=t,
        investigators=[
            Person(
                name="Jayaram Chandrashekar", registry_identifier="0000-0001-6412-0114"
            )
        ],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="MSMA Platform Development",
    )

    # Serialize back to JSON
    with open(output_json_path, "w") as f:
        f.write(data_description.model_dump_json(indent=3))

    logging.info(f"Validated and copied data_description.json to {output_json_path}")


def copy_processing(input_dir, output_dir):
    """Copy and validate processing metadata for SmartSPIM template."""
    input_json_path = input_dir / "processing.json"
    output_json_path = output_dir / "processing.json"

    with open(input_json_path, "r") as f:
        data = json.load(f)

    nps = len(data["processing_pipeline"]["data_processes"])

    # Load and validate the data with the Processing model
    processing = Processing(
        data_processes=[
            DataProcess(
                process_type=dp["name"],
                stage="Processing",
                experimenters=["Di Wang"],
                start_date_time=dp["start_date_time"],
                end_date_time=dp["end_date_time"],
                output_parameters=dp["outputs"],
                name=str(i),
                notes=dp["notes"],
                code=Code(
                    url=dp["code_url"],
                    version=dp["code_version"],
                    parameters=dp["parameters"],
                    input_data=[
                        DataAsset(url=s) for s in dp["input_location"].split(",")
                    ],
                ),
            )
            for i, dp in enumerate(data["processing_pipeline"]["data_processes"])
        ],
        dependency_graph={str(i): [str(i + 1)] for i in range(nps)},
    )

    # Serialize back to JSON
    with open(output_json_path, "w") as f:
        f.write(processing.model_dump_json(indent=3))

    logging.info(f"Validated and copied processing.json to {output_json_path}")


def package_smartspim_template(input_dir, results_dir, library, scales):
    """Complete SmartSPIM template packaging workflow.

    Args:
        input_dir (Path): Path to input directory containing SmartSPIM template data
        results_dir (Path): Path to results directory for output
        library (AssetLibrary): Asset library to register created assets
        scales (tuple): Scales to process (micrometers per voxel)
    """
    template = AnatomicalTemplate(
        name="allen-adult-mouse-spim-lca-template", version="2024-05", scales=scales
    )

    logging.info(f"Packaging SmartSPIM template {template.name}")
    logging.info(f"Processing scales: {template.scales}")

    # Create the anatomical template from SmartSPIM data
    template.create(
        input_prefix=input_dir / "smartspim_lca_template", output_root=results_dir
    )
    library.add(template)
    logging.info(
        f"Created SmartSPIM anatomical template {template.name} {template.version}"
    )

    # Copy validated metadata to template directory
    template_dir = template.location(results_dir)
    copy_data_description(input_dir, template_dir, template.name)
    copy_processing(input_dir, template_dir)

    # Create annotation sets with CCF labels mapped to SmartSPIM space
    create_smartspim_annotation_set(input_dir, results_dir, library)

    # Create coordinate transformations between SmartSPIM and CCF spaces
    create_smartspim_coordinate_transformations(input_dir, results_dir, library)

    logging.info(f"SmartSPIM template packaging complete for {template.name}")
