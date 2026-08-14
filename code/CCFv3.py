"""CCF 3 mouse brain atlas packaging functions."""

import logging
import shutil
from pathlib import Path

import pandas as pd

from atlas_builder import (
    AnnotationSet,
    CoordinateSpace,
    Template,
    Atlas,
    Terminology,
)
from atlas_builder.annotation_set import uncompress_annotations_to_zarr

import datetime
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema_models.data_name_patterns import build_data_name
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema.components.identifiers import Person


CCF2_TEMPLATE_CREATION_TIME = datetime.datetime(2012, 11, 27, tzinfo=datetime.timezone.utc)
CCF3_TEMPLATE_CREATION_TIME = datetime.datetime(2015, 4, 17, tzinfo=datetime.timezone.utc)
CCF3_2015_ANNOTATION_CREATION_TIME = datetime.datetime(2015, 4, 17, tzinfo=datetime.timezone.utc)
CCF3_2016_ANNOTATION_CREATION_TIME = datetime.datetime(2016, 5, 1, tzinfo=datetime.timezone.utc)
CCF3_2017_ANNOTATION_CREATION_TIME = datetime.datetime(2017, 5, 1, tzinfo=datetime.timezone.utc)
CCF3_ONTOLOGY_CREATION_TIME = datetime.datetime(2015, 4, 17, tzinfo=datetime.timezone.utc)

CCF2_TEMPLATE_SUMMARY = "This template is based upon the Allen Reference Atlas (ARA) specimen (Dong, 2008) in which a 3D volume was reconstructed using 528 Nissl sections of a near complete mouse brain. Approximately 200 structures were extracted from the 2-D atlas drawings to create 3-D annotations. This version (2011) supported the scientific objectives of the Allen Mouse Brain Connectivity Atlas (Oh et al., 2014) where a double-sided and more deeply annotated framework was needed. During the development, flaws in the 3D reconstructions were corrected and the volume was mirrored across the mid-line to create a symmetric space."
CCF3_TEMPLATE_SUMMARY = "The 2015 and initial release of the Allen Mouse Common Coordinate Framework anatomical template. The brain template was constructed as a population average of 1,675 young adult C57BL/6J mice brains imaged using serial two photon tomography (STPT) for the Allen Mouse Brain Connectivity Atlas (Oh et. al, 2014, Kuan et al, 2015). The average template was created from tissue autofluorescence detected in the red channel. To maximize input data and create a symmetrical atlas, each dataset was reflected across the midline, for a total of 3,350 (2 x 1,675) hemisphere datasets. Creation of the template followed a two-step iterative process: (1) We deformably registered each specimen to the current iteration of the template and computed an intensity average. (2) We then computed the average deformation field, inverted it, and applied it to the intensity average created in (1). This resulted in a volume with an average unbiased shape and intensity to be used as the template in the next iteration until convergence. In 10 µm resolution, the average template contains 506 million voxels with dimensions are 1,320 (X axis, anterior to posterior, 13.2 mm) x 1,140 (Y axis, left to right, 11.4 mm) x 800 (Z axis, dorsal to ventral, 8.0 mm). The origin of the coordinates is located at the the most anterior, left and dorsal corner of the volume. Due to constraint of the original data, the most rostral part of the main olfactory bulb and the most caudal parts of the medulla and cerebellum are excluded in the template."
CCF3_2015_ANNOTATION_DESCRIPTION = "The 2015 and initial release of the Allen Mouse Common Coordinate Framework, Annotation The process of parcellating the average template of the CCF is detailed in Wang et al, 2020. For any given structure, the process starts with a review of previously published atlases and literature and visual analyses of the average template and multimodal reference datasets. Data types include (1) transgenic expression data imaged with two-photon serial tomography, (2) axonal projection data from the Allen Mouse Connectivity Atlas, (3) immunohistochemical and (4) cytoarchitectural stains, including antibodies against NeuN, NF-160, SMI-32, parvalbumin, SMI-99, and calbindin, as well as stains for DAPI, Nissl, and AChE; and (5) in situ hybridization (ISH) gene expression data from the Allen Mouse Brain Atlas. Specific datasets used for the delineation of brain structures are listed in supplementary table Table S3 of Wang et. al., 2020. The format of the annotation is a 10 µm resolution image volume of the same size and orientation as the average brain template. Each voxel in the brain is labeled with a structure from the Allen Mouse Reference Atlas, Ontology. Voxels are annotated with the label for the most specific (finest) structure that it is a part of. It is inferred the voxel is also a part of any enclosing/parent structures as defined in the hierarchical tree of the ontology. The 2015 release includes 185 regions that were annotated directly in 3D using multimodal reference data. These newly drawn structures spanned 50% of the brain. To obtain full brain coverage, remaining areas were filled with annotations imported from the Allen Mouse Reference Atlas, creating a hybrid parcellation scheme. The interface between the old and new structures were manually adjusted to have smooth transitions."
CCF3_2016_ANNOTATION_DESCRIPTION = "The 2016 release of the Allen Mouse Common Coordinate Framework, Annotation The process of parcellating the average template of the CCF is detailed in Wang et al, 2020. For any given structure, the process starts with a review of previously published atlases and literature and visual analyses of the average template and multimodal reference datasets. Data types include (1) transgenic expression data imaged with two-photon serial tomography, (2) axonal projection data from the Allen Mouse Connectivity Atlas, (3) immunohistochemical and (4) cytoarchitectural stains, including antibodies against NeuN, NF-160, SMI-32, parvalbumin, SMI-99, and calbindin, as well as stains for DAPI, Nissl, and AChE; and (5) in situ hybridization (ISH) gene expression data from the Allen Mouse Brain Atlas. Specific datasets used for the delineation of brain structures are listed in supplementary table Table S3 of Wang et. al., 2020. The format of the annotation is a 10 µm resolution image volume of the same size and orientation as the average brain template. Each voxel in the brain is labeled with a structure from the Allen Mouse Reference Atlas, Ontology. Voxels are annotated with the label for the most specific (finest) structure that it is a part of. It is inferred the voxel is also a part of any enclosing/parent structures as defined in the hierarchical tree of the ontology. The 2016 update includes the complete 3D annotation of the isocortex, delineating 43 regions and 6 layers using gene expression and projection data and a novel curved cortical coordinates approach."
CCF3_2017_ANNOTATION_DESCRIPTION = "The 2017 release of the Allen Mouse Common Coordinate Framework, Annotation The process of parcellating the average template of the CCF is detailed in Wang et al, 2020. For any given structure, the process starts with a review of previously published atlases and literature and visual analyses of the average template and multimodal reference datasets. Data types include (1) transgenic expression data imaged with two-photon serial tomography, (2) axonal projection data from the Allen Mouse Connectivity Atlas, (3) immunohistochemical and (4) cytoarchitectural stains, including antibodies against NeuN, NF-160, SMI-32, parvalbumin, SMI-99, and calbindin, as well as stains for DAPI, Nissl, and AChE; and (5) in situ hybridization (ISH) gene expression data from the Allen Mouse Brain Atlas. Specific datasets used for the delineation of brain structures are listed in supplementary table Table S3 of Wang et. al., 2020. The format of the annotation is a 10 µm resolution image volume of the same size and orientation as the average brain template. Each voxel in the brain is labeled with a structure from the Allen Mouse Reference Atlas, Ontology. Voxels are annotated with the label for the most specific (finest) structure that it is a part of. It is inferred the voxel is also a part of any enclosing/parent structures as defined in the hierarchical tree of the ontology. In the 2017 release, the parcellation spanned 43 isocortical areas and their layers, 329 subcortical gray matter structures, 81 fiber tracts, and 8 ventricular structures."
CCF3_ONTOLOGY_DESCRIPTION = "The 2017 release of the Allen Mouse Reference Atlas Ontology. The Allen Mouse Reference Atlas Ontology defines a hierarchical partonomy of the anatomical structures of the adult mouse brain. At the top level, the brain is divided into gray matter, fiber tracts and ventricular systems. Gray matter is subdivided into three large regions (cerebrum, brain stem, and cerebellum), which are themselves organized into subregions in a hierarchical tree. The Allen Mouse Reference Atlas, Ontology was developed for the Allen Reference Atlas (Dong, 2008) and follows terminology from “Brain Maps: Structure for the Rat Brain” (Swanson, 2004, 2018). The ontology has been subsequently extended and revised to also serve as the structure ontology for the Allen Mouse Common Coordinate Framework (Wang et al, 2020). This 2017 release of the Allen Mouse Reference Atlas Ontology is in support of the 2017 Release of the Allen Mouse Common Framework."


def _write_template_data_description(
    output_dir: Path, name: str, version: str, summary: str, modalities, creation_time
):
    """Create and write a data_description.json for a template using AIND schema.

    Args:
        output_dir: Directory of the anatomical template where JSON will be written.
        name: Asset name (e.g., "allen-adult-mouse-stpt-template").
        summary: Long-form data summary text.
        modalities: List of Modality enums describing the imaging modalities.
        creation_time: Datetime for creation_time (required).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dd = DataDescription(
        name=build_data_name(name.replace(".", "-"), creation_time),
        data_summary=summary.strip(),
        subject_id="adult-mouse-population-average",
        modalities=modalities,
        data_level="derived",
        creation_time=creation_time,
        institution=Organization.AIBS,
        investigators=[Person(name="Lydia Ng", registry_identifier="0000-0002-7499-3514")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )

    dd.write_standard_file(output_directory=output_dir)

    logging.info(f"Wrote data_description.json for {name} to {output_dir}")


def _write_annotation_data_description(output_dir: Path, name: str, version: str, summary: str, creation_time):
    """Create and write a data_description.json for an anatomical annotation set.

    Args:
        output_dir: Directory of the annotation set where JSON will be written.
        name: Asset name (e.g., "allen-adult-mouse-annotation").
        version: Version string (e.g., "2017").
        summary: Long-form data summary text for the annotation release.
        creation_time: Datetime for creation_time (required).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dd = DataDescription(
        name=build_data_name(f"{name.replace('.', '-')}-{version}", creation_time),
        data_summary=summary.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],  # Derived from STPT population template
        data_level="derived",
        creation_time=creation_time,
        institution=Organization.AIBS,
        investigators=[Person(name="Quanxin Wang", registry_identifier="0000-0002-0007-7935")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )

    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for annotation set {name} {version} to {output_dir}")


def _write_ontology_data_description(output_dir: Path, name: str, version: str, summary: str, creation_time):
    """Create and write a data_description.json for the ontology (terminology)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name(f"{name.replace('.', '-')}-{version}", creation_time),
        data_summary=summary.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],  # Derived from multimodal data incl. STPT
        data_level="derived",
        creation_time=creation_time,
        institution=Organization.AIBS,
        investigators=[Person(name="Lydia Ng", registry_identifier="0000-0002-7499-3514")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for ontology {name} {version} to {output_dir}")


def create_all_ccf_templates(
    input_dir, results_dir, library, scales=(10, 25, 50, 100)
):
    """Create templates from CCF 3 atlas data."""
    # Create 2-photon average template
    average_template_prefix = input_dir / "average_template" / "average_template"
    template = Template(
        name="allen-adult-mouse-stpt-template", version="2015", scales=scales
    )
    template.create(average_template_prefix, results_dir)
    library.add(template)
    logging.info(f"Created average_template: {template.name} {template.version}")

    # Write data description for the CCFv3 two-photon average template
    _write_template_data_description(
        output_dir=template.location(results_dir),
        name=template.name,
        version=template.version,
        summary=CCF3_TEMPLATE_SUMMARY,
        modalities=[Modality.STPT],
        creation_time=CCF3_TEMPLATE_CREATION_TIME,
    )

    # Create Nissl reference template
    ara_nissl_prefix = input_dir / "ara_nissl" / "ara_nissl"
    template = Template(
        name="allen-adult-mouse-nissl-template", version="2011", scales=scales
    )
    template.create(ara_nissl_prefix, results_dir)
    library.add(template)
    logging.info(f"Created ara_nissl: {template.name} {template.version}")

    # Write data description for the ARA Nissl (2011) template
    _write_template_data_description(
        output_dir=template.location(results_dir),
        name=template.name,
        version=template.version,
        summary=CCF2_TEMPLATE_SUMMARY,
        modalities=[Modality.BRIGHTFIELD],
        creation_time=CCF2_TEMPLATE_CREATION_TIME,
    )


def create_all_ccf_annotation_sets(input_dir, results_dir, library, scales=(10, 25, 50, 100)):
    """Create all CCF anatomical annotation sets across different versions and templates."""
    logging.info("Creating all CCF anatomical annotation sets...")

    # Get templates and terminology from library
    template_stpt = library.get_template(
        "allen-adult-mouse-stpt-template", "2015"
    )
    template_nissl = library.get_template(
        "allen-adult-mouse-nissl-template", "2011"
    )
    terminology = library.get_terminology(
        "allen-adult-mouse-terminology", "2017"
    )
    devmouse_terminology = library.get_terminology(
        "allen-dev-mouse-terminology", "2012"
    )

    # Define annotation configurations for different CCF versions
    annotations = [
        {
            "directory": "ccf_2015",
            "template": template_stpt,
            "version": "2015",
            "name": "allen-adult-mouse-annotation",
            "summary": CCF3_2015_ANNOTATION_DESCRIPTION,
            "creation_time": CCF3_2015_ANNOTATION_CREATION_TIME,
            "terminology": terminology,
        },
        {
            "directory": "ccf_2016",
            "template": template_stpt,
            "version": "2016",
            "name": "allen-adult-mouse-annotation",
            "summary": CCF3_2016_ANNOTATION_DESCRIPTION,
            "creation_time": CCF3_2016_ANNOTATION_CREATION_TIME,
            "terminology": terminology,
        },
        {
            "directory": "ccf_2017",
            "template": template_stpt,
            "version": "2017",
            "name": "allen-adult-mouse-annotation",
            "summary": CCF3_2017_ANNOTATION_DESCRIPTION,
            "creation_time": CCF3_2017_ANNOTATION_CREATION_TIME,
            "terminology": terminology,
        },
        {
            "directory": "devmouse_2012",
            "template": template_nissl,
            "version": "2012",
            "name": "allen-dev-mouse-p56-nissl-annotation",
            "summary": None,
            "creation_time": None,
            "terminology": devmouse_terminology,
        },
        {
            "directory": "mouse_2011",
            "template": template_nissl,
            "version": "2011",
            "name": "allen-adult-mouse-annotation",
            # No data description requested for 2011 in this change
            "summary": None,
            "creation_time": None,
            "terminology": terminology,
        },
    ]

    for annotation in annotations:
        annotation_dir = input_dir / "annotation" / annotation["directory"]
        annotation_set = AnnotationSet(
            name=annotation["name"],
            coordinate_space=annotation["template"].coordinate_space,
            template=annotation["template"],
            terminology=annotation["terminology"],
            version=annotation["version"],
            scales=scales,
        )

        annotation_set.create_from_nifti(
            input_prefix=annotation_dir / "annotation",
            output_root=results_dir,
            # Only 2017 has ever carried meshes; meshing every set would be new output
            # and a meshing pass per hierarchy level on each of them.
            include_meshes=annotation["version"] == "2017",
        )

        annotation_output_dir = annotation_set.location(results_dir)
        uncompress_annotations_to_zarr(
            input_dir=annotation_output_dir,
            terminology=terminology,
            output_dir=annotation_output_dir,
            scales=annotation_set.scales,
        )

        # Write data description only for specified CCF 2015-2017 annotation sets
        if annotation["summary"] is not None and annotation["creation_time"] is not None:
            _write_annotation_data_description(
                output_dir=annotation_set.location(results_dir),
                name=annotation_set.name,
                version=annotation_set.version,
                summary=annotation["summary"],
                creation_time=annotation["creation_time"],
            )

        annotation_set.create_manifest(results_dir)
        library.add(annotation_set)

    logging.info("All CCF anatomical annotation sets created successfully")


def create_ccf3_terminology(input_dir, output_dir, library):
    """Create parcellation terminology from CCF 3 structure hierarchy."""
    input_path = input_dir / "annotation" / "adult_mouse_ccf_structures.csv"

    df = pd.read_csv(input_path)

    # Create DataFrame with required columns for Terminology
    # For CCF3, use structure IDs as both file_id and identifier
    filtered_df = pd.DataFrame(
        {
            "identifier": df["id"].map(lambda x: f"MBA:{int(x)}"),
            "annotation_value": df["id"].astype(int),
            "parent_identifier": df["parent_structure_id"].map(lambda x: f"MBA:{int(x)}" if not pd.isna(x) else ""),
            "name": df["name"],
            "abbreviation": df["acronym"],
            "color_hex_triplet": df["color_hex_triplet"].map(lambda x: f"#{x}"),
        }
    )

    terminology = Terminology(
        name="allen-adult-mouse-terminology", version="2017", df=filtered_df
    )

    # Build identifier -> annotation_value lookup since identifiers are prefixed
    id_to_ann = dict(zip(terminology.df["identifier"], terminology.df["annotation_value"]))
    terminology.set_descendant_annotation_values(
        lambda row: [id_to_ann[i] for i in row["descendant_identifiers"] if i in id_to_ann]
    )

    parcellation_legacy_dir = terminology.location(output_dir) / "legacy_files"
    parcellation_legacy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, parcellation_legacy_dir / input_path.name)

    terminology.write_terminology(output_dir)
    terminology.create_manifest(output_dir)
    # Write ontology data description
    _write_ontology_data_description(
        output_dir=terminology.location(output_dir),
        name=terminology.name,
        version=terminology.version,
        summary=CCF3_ONTOLOGY_DESCRIPTION,
        creation_time=CCF3_ONTOLOGY_CREATION_TIME,
    )
    library.add(terminology)


def package_ccf(input_dir, output_dir, library, scales=(10, 25, 50, 100)):
    """Complete packaging workflow for CCF 3 atlas data."""
    # Create and register terminologies
    create_ccf3_terminology(input_dir, output_dir, library)

    # Create and register anatomical templates
    create_all_ccf_templates(input_dir, output_dir, library, scales)

    # Create and register coordinate spaces
    coordinate_space = CoordinateSpace(
        name="allen-adult-mouse-ccf-space",
        version="2015",
    )
    coordinate_space.create_manifest(output_dir)
    library.add(coordinate_space)

    library.get_template("allen-adult-mouse-stpt-template", "2015").coordinate_space = coordinate_space

    coordinate_space = CoordinateSpace(
        name="allen-adult-mouse-ccf-space",
        version="2011",
    )
    coordinate_space.create_manifest(output_dir)
    library.add(coordinate_space)

    library.get_template("allen-adult-mouse-nissl-template", "2011").coordinate_space = coordinate_space

    # Create and register annotation sets
    create_all_ccf_annotation_sets(input_dir, output_dir, library, scales=scales)

    # Create and register parcellation atlas
    atlases = [
        Atlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2011",
            coordinate_space=library.get_coordinate_space(
                name="allen-adult-mouse-ccf-space", version="2011"
            ),
            templates=[library.get_template(
                name="allen-adult-mouse-nissl-template", version="2011"
            )],
            annotation_sets=[library.get_annotation_set(
                name="allen-adult-mouse-annotation", version="2011"
            )],
        ),
        Atlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2015",
            coordinate_space=library.get_coordinate_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            templates=[library.get_template(
                name="allen-adult-mouse-stpt-template", version="2015"
            )],
            annotation_sets=[library.get_annotation_set(
                name="allen-adult-mouse-annotation", version="2015"
            )],
        ),
        Atlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2016",
            coordinate_space=library.get_coordinate_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            templates=[library.get_template(
                name="allen-adult-mouse-stpt-template", version="2015"
            )],
            annotation_sets=[library.get_annotation_set(
                name="allen-adult-mouse-annotation", version="2016"
            )],
        ),
        Atlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2017",
            coordinate_space=library.get_coordinate_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            templates=[library.get_template(
                name="allen-adult-mouse-stpt-template", version="2015"
            )],
            annotation_sets=[library.get_annotation_set(
                name="allen-adult-mouse-annotation", version="2017"
            )],
        ),
        Atlas(
            name="allen-dev-mouse-p56-atlas",
            version="2012",
            coordinate_space=library.get_coordinate_space(
                name="allen-adult-mouse-ccf-space", version="2011"
            ),
            templates=[library.get_template(
                name="allen-adult-mouse-nissl-template", version="2011"
            )],
            annotation_sets=[library.get_annotation_set(
                name="allen-dev-mouse-p56-nissl-annotation", version="2012"
            )],
        ),
    ]
    for atlas in atlases:
        atlas.create_manifest(output_dir)
        library.add(atlas)
