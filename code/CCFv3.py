"""CCF 3 mouse brain atlas packaging functions."""

import glob
import logging
import shutil
from pathlib import Path

import pandas as pd

from atlas_assets import (AnatomicalAnnotationSet, AnatomicalSpace,
                          AnatomicalTemplate, ParcellationAtlas,
                          ParcellationTerminology)
from atlas_assets.mesh import Mesh
from atlas_assets.precomputed import append_meshes_to_precomputed

# Added imports for data description generation
import datetime
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema_models.data_name_patterns import build_data_name
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema.components.identifiers import Person


# Placeholder creation times to be replaced later (distinct per template)
CCF2_TEMPLATE_CREATION_TIME = datetime.datetime(2012, 11, 27, tzinfo=datetime.timezone.utc)
CCF3_TEMPLATE_CREATION_TIME = datetime.datetime(2015, 4, 17, tzinfo=datetime.timezone.utc)

CCF2_TEMPLATE_SUMMARY = """
This template is based upon the Allen Reference Atlas (ARA) specimen (Dong, 2008) in which a 3D volume was
reconstructed using 528 Nissl sections of a near complete mouse brain. Approximately 200 structures were extracted
from the 2-D atlas drawings to create 3-D annotations. This version (2011) supported the scientific objectives
of the Allen Mouse Brain Connectivity Atlas (Oh et al., 2014) where a double-sided and more deeply
annotated framework was needed. During the development, flaws in the 3D reconstructions were
corrected and the volume was mirrored across the mid-line to create a symmetric space.
"""

CCF3_TEMPLATE_SUMMARY = """
This is a 3D spatial template constructed as a population average of 1,675 young adult mouse brains imaged using
serial two photon tomography for the Allen Mouse Brain Connectivity Atlas (Kuan et al., 2015, Oh et al., 2014,
Ragan et al., 2012). The average was created from tissue autofluorescence detected in the red channel. To maximize
input data and create a symmetrical atlas, each hemisphere was reflected across the midline, for a total of 3,350
image series (= 2 × 1,675 brains). Images were acquired at high resolution (x, y = 0.35 μm/pixel) every 100 μm
through the anterior-posterior axis of the brain, then downsampled to 50, 25, and 10 μm in x-y axes. Slight offsets
in the position where imaging starts for each brain provide sufficient coverage to allow interpolation along the z axis
to obtain isotropic voxel resolution to 10 μm. Assuming uniform sampling along the z axis, each 10 μm is spanned by data
from 335 hemispheres, a number comparable to Fonov et al. (2011). We started with a previous template (Kuan et al., 2015,
Oh et al., 2014), adding more registered brains to create the new CCFv3 initial template. An iterative process (1)
deformably registered each specimen to the template and averaged all specimens, and (2) computed the average deformation
field over all specimens, then inverted and applied it to the average image created in (1). This resulted in a volume
with an average unbiased shape and intensity used as the template in the next iteration. The algorithm continued until
the difference between the mean magnitude of the average deformation field between iterations fell below a certain
threshold and stabilized. Figure 1A illustrates the convergence to a sharp average image with evident anatomical details.
The average template was matched in size and anterior-posterior extent to the Allen Reference Atlas Nissl-stained specimen 
(allen-adult-mouse-nissl-tempalte) to retain integrity with the original coordinates and dependent informatic tools.
The most rostral part of the main olfactory bulb and the most caudal parts of the medulla and cerebellum are excluded.
At 10 μm voxel resolution, the average template contains ~506 million voxels. Its dimensions are 1,320 (anterior to posterior,
13.2 mm) × 1,140 (left to right, 11.4 mm) × 800 voxels (dorsal to ventral, 8.0 mm).
"""


def _write_template_data_description(output_dir: Path, name: str, version: str, summary: str, modalities, creation_time):
    """Create and write a data_description.json for a template using AIND schema.

    Args:
        output_dir: Directory of the anatomical template where JSON will be written.
        name: Asset name (e.g., "allen-adult-mouse-2p-template").
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
        institution=Organization.AIND,
        investigators=[Person(name="Lydia Ng", registry_identifier="0000-0002-7499-3514")], 
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )

    with open(output_dir / "data_description.json", "w") as f:
        f.write(dd.model_dump_json(indent=3))

    logging.info(f"Wrote data_description.json for {name} to {output_dir}")


def load_ccf3_meshes(mesh_dir):
    """Load CCF 3 meshes from a directory.
    Args:
        mesh_dir (Path): Directory containing .obj mesh files.
    Yields:
        tuple: (Mesh, identifier) where Mesh is an instance of Mesh class and identifier is the mapped identifier.
    """

    for fname in glob.glob(str(mesh_dir / "*.obj")):
        obj_id = int(Path(fname).stem.split("_")[1])

        mesh = Mesh.from_obj(fname)
        yield mesh, obj_id


def create_all_ccf_anatomical_templates(
    input_dir, results_dir, library, scales=(10, 25, 50, 100)
):
    """Create anatomical templates from CCF 3 atlas data."""
    # Create 2-photon average template
    average_template_prefix = input_dir / "average_template" / "average_template"
    template = AnatomicalTemplate(
        name="allen-adult-mouse-2p-template", version="2015", scales=scales
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
    template = AnatomicalTemplate(
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


def create_all_ccf_annotation_sets(
    input_dir, results_dir, library, scales=(10, 25, 50, 100)
):
    """Create all CCF anatomical annotation sets across different versions and templates."""
    logging.info("Creating all CCF anatomical annotation sets...")

    # Get templates and terminology from library
    template_2p = library.get_anatomical_template(
        "allen-adult-mouse-2p-template", "2015"
    )
    template_nissl = library.get_anatomical_template(
        "allen-adult-mouse-nissl-template", "2011"
    )
    terminology = library.get_parcellation_terminology(
        "allen-adult-mouse-terminology", "2017"
    )

    # Define annotation configurations for different CCF versions
    annotations = [
        {
            "directory": "ccf_2015",
            "template": template_2p,
            "version": "2015",
            "name": "allen-adult-mouse-annotation",
        },
        {
            "directory": "ccf_2016",
            "template": template_2p,
            "version": "2016",
            "name": "allen-adult-mouse-annotation",
        },
        {
            "directory": "ccf_2017",
            "template": template_2p,
            "version": "2017",
            "name": "allen-adult-mouse-annotation",
        },
        {
            "directory": "devmouse_2012",
            "template": template_nissl,
            "version": "2012",
            "name": "allen-dev-mouse-p56-annotation",
        },
        {
            "directory": "mouse_2011",
            "template": template_nissl,
            "version": "2011",
            "name": "allen-adult-mouse-annotation",
        },
    ]

    for annotation in annotations:
        annotation_dir = input_dir / "annotation" / annotation["directory"]
        annotation_set = AnatomicalAnnotationSet(
            name=annotation["name"],
            anatomical_template=annotation["template"],
            parcellation_terminology=terminology,
            version=annotation["version"],
            scales=scales,
        )

        annotation_set.create_from_nifti(
            input_prefix=annotation_dir / "annotation",
            output_root=results_dir,
        )

        annotation_set.create_manifest(results_dir)
        library.add(annotation_set)

    meshes = load_ccf3_meshes(
        Path("/data/ccf_meshes/mcc/annotation/ccf_2017/structure_meshes")
    )
    append_meshes_to_precomputed(
        meshes,
        results_dir
        / "anatomical-annotation-sets"
        / "allen-adult-mouse-annotation"
        / "2017"
        / "annotations.precomputed",
        scale=1000,  # convert to nanometers
    )

    logging.info("All CCF anatomical annotation sets created successfully")


def create_ccf3_parcellation_terminology(input_dir, output_dir, library):
    """Create parcellation terminology from CCF 3 structure hierarchy."""
    input_path = input_dir / "annotation" / "adult_mouse_ccf_structures.csv"

    df = pd.read_csv(input_path)

    # Create DataFrame with required columns for ParcellationTerminology
    # For CCF3, use structure IDs as both file_id and identifier
    filtered_df = pd.DataFrame(
        {
            "identifier": df["id"].map(lambda x: f"MBA:{int(x)}"),
            "annotation_value": df["id"].astype(int),
            "parent_identifier": df["parent_structure_id"].map(
                lambda x: f"MBA:{int(x)}" if not pd.isna(x) else ""
            ),
            "name": df["name"],
            "abbreviation": df["acronym"],
            "color_hex_triplet": df["color_hex_triplet"].map(lambda x: f"#{x}"),
        }
    )

    terminology = ParcellationTerminology(
        name="allen-adult-mouse-terminology", version="2017", df=filtered_df
    )

    # Build identifier -> annotation_value lookup since identifiers are prefixed
    id_to_ann = dict(
        zip(terminology.df["identifier"], terminology.df["annotation_value"])
    )
    terminology.set_descendant_annotation_values(
        lambda row: [id_to_ann[i] for i in row["descendants"] if i in id_to_ann]
    )

    parcellation_legacy_dir = terminology.location(output_dir) / "legacy_files"
    parcellation_legacy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, parcellation_legacy_dir / input_path.name)

    terminology.write_terminology(output_dir)
    terminology.create_manifest(output_dir)
    library.add(terminology)


def package_ccf(input_dir, output_dir, library, scales=(10, 25, 50, 100)):
    """Complete packaging workflow for CCF 3 atlas data."""
    # Create and register terminologies
    create_ccf3_parcellation_terminology(input_dir, output_dir, library)

    # Create and register anatomical templates
    create_all_ccf_anatomical_templates(input_dir, output_dir, library, scales)

    # Create and register annotation sets
    create_all_ccf_annotation_sets(input_dir, output_dir, library, scales=scales)

    # Create and register anatomical space
    anatomical_space = AnatomicalSpace(
        name="allen-adult-mouse-ccf-space",
        version="2015",
        anatomical_template=library.get_anatomical_template(
            "allen-adult-mouse-2p-template", "2015"
        ),
    )
    library.add(anatomical_space)

    anatomical_space = AnatomicalSpace(
        name="allen-adult-mouse-ccf-space",
        version="2011",
        anatomical_template=library.get_anatomical_template(
            "allen-adult-mouse-nissl-template", "2011"
        ),
    )
    library.add(anatomical_space)

    # Create and register parcellation atlas
    atlases = [
        ParcellationAtlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2011",
            anatomical_space=library.get_anatomical_space(
                name="allen-adult-mouse-ccf-space", version="2011"
            ),
            anatomical_annotation_set=library.get_anatomical_annotation_set(
                name="allen-adult-mouse-annotation", version="2011"
            ),
            parcellation_terminology=library.get_parcellation_terminology(
                name="allen-adult-mouse-terminology", version="2017"
            ),
        ),
        ParcellationAtlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2015",
            anatomical_space=library.get_anatomical_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            anatomical_annotation_set=library.get_anatomical_annotation_set(
                name="allen-adult-mouse-annotation", version="2015"
            ),
            parcellation_terminology=library.get_parcellation_terminology(
                name="allen-adult-mouse-terminology", version="2017"
            ),
        ),
        ParcellationAtlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2016",
            anatomical_space=library.get_anatomical_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            anatomical_annotation_set=library.get_anatomical_annotation_set(
                name="allen-adult-mouse-annotation", version="2016"
            ),
            parcellation_terminology=library.get_parcellation_terminology(
                name="allen-adult-mouse-terminology", version="2017"
            ),
        ),
        ParcellationAtlas(
            name="allen-adult-mouse-ccf-atlas",
            version="2017",
            anatomical_space=library.get_anatomical_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            anatomical_annotation_set=library.get_anatomical_annotation_set(
                name="allen-adult-mouse-annotation", version="2017"
            ),
            parcellation_terminology=library.get_parcellation_terminology(
                name="allen-adult-mouse-terminology", version="2017"
            ),
        ),
        ParcellationAtlas(
            name="allen-dev-mouse-p56-atlas",
            version="2012",
            anatomical_space=library.get_anatomical_space(
                name="allen-adult-mouse-ccf-space", version="2015"
            ),
            anatomical_annotation_set=library.get_anatomical_annotation_set(
                name="allen-dev-mouse-p56-annotation", version="2012"
            ),
            parcellation_terminology=library.get_parcellation_terminology(
                name="allen-dev-mouse-terminology", version="2012"
            ),
        ),
    ]
    for atlas in atlases:
        atlas.create_manifest(output_dir)
        library.add(atlas)
