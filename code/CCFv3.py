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

    # Create Nissl reference template
    ara_nissl_prefix = input_dir / "ara_nissl" / "ara_nissl"
    template = AnatomicalTemplate(
        name="allen-adult-mouse-nissl-template", version="2011", scales=scales
    )
    template.create(ara_nissl_prefix, results_dir)
    library.add(template)
    logging.info(f"Created ara_nissl: {template.name} {template.version}")


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
