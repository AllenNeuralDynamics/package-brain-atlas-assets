"""Allen CCF 2020 atlas packaging from ABC Atlas data."""

import logging
import shutil
from pathlib import Path

import pandas as pd
from CCFv3 import load_ccf3_meshes

from atlas_builder import (AnnotationSet, CoordinateSpace,
                          Template, Atlas,
                          Terminology)
from atlas_builder.precomputed import append_meshes_to_precomputed, clear_meshes_from_precomputed
from atlas_builder.annotation_set import uncompress_annotations_to_zarr
import datetime
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema_models.data_name_patterns import build_data_name
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema.components.identifiers import Person

CCF2020_MINTED_VALUE_OFFSET = 2000
CCF2020_LABEL_REFERENCE = (
    "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2022/compacted/itksnap_label_description.txt"
)


CCF2020_TERMINOLOGY_CREATION_TIME = datetime.datetime(2020, 4, 17, tzinfo=datetime.timezone.utc)
CCF2020_ANNOTATION_CREATION_TIME = datetime.datetime(2020, 4, 17, tzinfo=datetime.timezone.utc)
CCF2020_TEMPLATE_CREATION_TIME = datetime.datetime(2020, 4, 17, tzinfo=datetime.timezone.utc)

CCF2020_ANNOTATION_DESCRIPTION = "The 2020 release of the Allen Mouse Common Coordinate Framework, Annotation The process of parcellating the average template of the CCF is detailed in Wang et al, 2020. For any given structure, the process starts with a review of previously published atlases and literature and visual analyses of the average template and multimodal reference datasets. Data types include (1) transgenic expression data imaged with two-photon serial tomography, (2) axonal projection data from the Allen Mouse Connectivity Atlas, (3) immunohistochemical and (4) cytoarchitectural stains, including antibodies against NeuN, NF-160, SMI-32, parvalbumin, SMI-99, and calbindin, as well as stains for DAPI, Nissl, and AChE; and (5) in situ hybridization (ISH) gene expression data from the Allen Mouse Brain Atlas. Specific datasets used for the delineation of brain structures are listed in supplementary table Table S3 of Wang et. al., 2020. The format of the annotation is a 10 µm resolution image volume of the same size and orientation as the average brain template. Each voxel in the brain is labeled with a structure from the Allen Mouse Reference Atlas, Ontology. Voxels are annotated with the label for the most specific (finest) structure that it is a part of. It is inferred the voxel is also a part of any enclosing/parent structures as defined in the hierarchical tree of the ontology. The 2020 release adds new annotations for layers of the Ammon’s horn (CA), main olfactory bulb (MOB) and minor modification of surrounding fiber tracts. With this release, the origin of the volume has been moved to the anterior commissure to align with the origin of the developing mouse templates."
CCF2020_TERMINOLOGY_DESCRIPTION = "The 2020 release of the Allen Mouse Reference Atlas, Ontology. The Allen Mouse Reference Atlas, Ontology defines a hierarchical partonomy of the anatomical structures of the adult mouse brain. At the top level, the brain is divided into gray matter, fiber tracts and ventricular systems. Gray matter is subdivided into three large regions (cerebrum, brain stem, and cerebellum), which are themselves organized into subregions in a hierarchical tree. The Allen Mouse Reference Atlas, Ontology was developed for the Allen Reference Atlas (Dong, 2008) and follows terminology from “Brain Maps: Structure for the Rat Brain” (Swanson, 2004, 2018). The ontology has been subsequently extended and revised to also serve as the structure ontology for the Allen Mouse Common Coordinate Framework (Wang et al, 2020). The 2020 release introduces a new concept of a 'term set', which is a collection of parcellation terms that share a common set of properties, in this case organizing the ontology in 'organ', 'category', 'division', 'structure', and 'substructure' levels. This release also changed the identifier scheme of the ontology, separating the annotation index from a new string-based label, to allow for more compact data types to be used for annotation. The 2020 onotology was produced to support the relesae of the Allen Brain Cell Atlas."
CCF2020_TEMPLATE_DESCRIPTION = "The 2020 release of the Allen Mouse Common Coordinate Framework, Template. This serial two-photon tomography template is identical to CCF 2017 template and coordinate space except for a change in the origin, which has been moved to the anterior commissure to align with the origin of the developing mouse templates. Voxel size, direction, and total size are not changed from CCF 2017."


def create_ccf2020_template(input_dir, results_dir, library, scales=(10,25)):
    """Create CCF 2020 anatomical template from ABC Atlas data."""
    logging.info("Creating CCF 2020 anatomical template...")

    # Create anatomical template from the ABC Atlas average template
    template_dir = input_dir / "image_volumes" / "Allen-CCF-2020" / "20250331"
    template = Template(
        name="allen-adult-mouse-stpt-template", version="2020", scales=scales
    )

    # Use the average template from ABC Atlas
    template.create(template_dir / "average_template", results_dir)
    library.add(template)
    # Write data description for template
    _write_ccf2020_template_data_description(template.location(results_dir))
    logging.info(f"Created CCF 2020 template: {template.name} {template.version}")

    return template


def _register_annotation_values(pt_df: pd.DataFrame, parcellation_df: pd.DataFrame) -> pd.Series:
    """Create an annotation value for every term in `parcellation.csv` that has no parcellation_index of its own.

    Uses the public ITK label table from 2020 release as a reference. Registers every value that has already
    been published, so a term keeps its value across builds. Anything that release does not cover is
    numbered `CCF2020_MINTED_VALUE_OFFSET + int(graph_order)`
    """
    real_values = set(pd.to_numeric(parcellation_df["parcellation_index"], errors="coerce").dropna().astype(int))

    # Read in ITK-SNAP label description file formatted as [index R G B A VIS MSH "<abbreviation> - <identifier>"]. Only need the first and last columns here.

    published_df = pd.read_table(
        CCF2020_LABEL_REFERENCE,
        sep=r"\s+",
        header=None,
        quotechar='"',
        usecols=[0, 7],
        names=["annotation_value", "label"],
    )
    
    # Identifier in ITK-SNAP label has stripped `MBA:`. Need to put back in for joining
    _mba_identifier = [f"MBA:{n.rsplit(' - ', 1)[-1].strip()}" for n in published_df["label"]]
    
    published = dict(
        zip(
            _mba_identifier,
            published_df["annotation_value"].astype(int),
        )
    )

    missing = pt_df.loc[pt_df["parcellation_index"].isna()]
    identifiers = missing["identifier"].astype(str)
    order = pd.to_numeric(missing["graph_order"], errors="coerce")
    if order.isna().any():
        raise ValueError(
            f"graph_order is missing for {missing.loc[order.isna(), 'identifier'].tolist()[:5]}; "
            "it is required to assign annotation values."
        )

    # Only terms with no parcellation_index take a value from reference (assumes rest is consistent w/ parcellation.csv).
    pinned = {ident: published[ident] for ident in set(identifiers) & set(published)}

    highest_pinned = max([0, *real_values, *pinned.values()])
    
    # Hardcode errors if annotation values run into each other

    if reused := sorted(set(pinned.values()) & real_values):
        raise ValueError(
            f"Last stable version gives voxel-less terms annotation values {reused[:5]} that the "
            f"annotation now uses, total: {len(reused)})."
        )

    values = identifiers.map(pinned).fillna(CCF2020_MINTED_VALUE_OFFSET + order.astype(int)).astype(int)

    pairs = pd.DataFrame({"identifier": identifiers, "value": values}).drop_duplicates()
    if clashes := sorted(pairs.loc[pairs.duplicated("value", keep=False), "value"].unique()):
        raise ValueError(
            f"Multiple independent terms share annotation values {clashes[:5]} ({len(clashes)} total)!"
        )

    unpublished = sorted(set(identifiers) - set(published))
    if unpublished:
        logging.warning(
            f"{len(unpublished)} terms are new: {unpublished[:10]}."
        )
    logging.info(
        f"Assigned {pairs['identifier'].nunique()} annotation values ({values.min()}-{values.max()}) to terms "
        f"with no parcellation_index of their own; {len(published)} pinned, {len(unpublished)} created"
    )
    return values


def _build_ccf2020_terminology_dataframe(metadata_dir: Path) -> pd.DataFrame:
    """Build the CCF 2020 terminology DataFrame from metadata CSVs."""
    # Load inputs
    pt_df = pd.read_csv(metadata_dir / "parcellation_term.csv")
    pptm_df = pd.read_csv(metadata_dir / "parcellation_to_parcellation_term_membership.csv")

    # --- DataFrame construction logic ported from code/ccf2020term.py ---
    # Load the remaining metadata tables used for enrichment.
    parcellation_df = pd.read_csv(metadata_dir / "parcellation.csv")
    ptsm_df = pd.read_csv(metadata_dir / "parcellation_term_set_membership.csv")
    pts_df = pd.read_csv(metadata_dir / "parcellation_term_set.csv")

    def _to_annotation_label(identifier):
        if isinstance(identifier, str):
            suffix = identifier.split(":", 1)[-1]
            # parcellation.csv label convention includes the year segment
            return f"AllenCCF-Annotation-2020-{suffix}"
        return pd.NA

    pt_df = pt_df.copy()
    pt_df["annotation_label"] = pt_df["identifier"].apply(_to_annotation_label)

    # Pull term set membership onto pt_df
    pt_df = pt_df.merge(
        ptsm_df[["parcellation_term_label", "parcellation_term_set_label"]].drop_duplicates(),
        how="left",
        left_on="label",
        right_on="parcellation_term_label",
    )
    pt_df = pt_df.drop(columns=["parcellation_term_label"])

    # Replace term set *labels* with their human-readable term set *names*
    # (e.g. AllenCCF-Ontology-2017-ORGA -> organ)
    pt_df = pt_df.merge(
        pts_df[["label", "name"]].rename(
            columns={"label": "parcellation_term_set_label", "name": "term_set_name"}
        ),
        how="left",
        on="parcellation_term_set_label",
    )
    pt_df = pt_df.drop(columns=["parcellation_term_set_label"])

    # Pull `parcellation_index` onto pt_df (matches are via the derived annotation_label)
    pt_df = pt_df.merge(
        parcellation_df[["label", "parcellation_index"]].rename(columns={"label": "annotation_label"}),
        on="annotation_label",
        how="left",
    )

    def _collapse_abc_unassigned(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse the 5 ABC-Ontology-2023-unassigned-* rows into a single row."""
        label_s = df["label"].astype(str)
        unassigned_mask = label_s.str.startswith("ABC-Ontology-2023-unassigned-", na=False)

        unassigned = df[unassigned_mask].copy()
        others = df[~unassigned_mask].copy()

        # If the source data doesn't contain these rows, do nothing.
        if unassigned.empty:
            return df

        # Template row: start from the first unassigned row to preserve expected columns
        row = unassigned.iloc[0].copy()
        row["label"] = "ABC-Ontology-2023-unassigned"
        row["acronym"] = "unassigned"
        row["name"] = "unassigned"
        row["identifier"] = "MBA:0"
        row["parcellation_index"] = 0
        row["term_set_name"] = sorted(
            set(
                unassigned["term_set_name"]
                .dropna()
                .astype(str)
                .loc[lambda s: s.str.strip() != ""]
                .tolist()
            )
        )
        row["annotation_label"] = pd.NA

        collapsed = pd.DataFrame([row])
        return pd.concat([others, collapsed], ignore_index=True)

    pt_df = _collapse_abc_unassigned(pt_df)

    # Fill in parcellation_index for any rows missing one using MBA identifier as invariant across build runs.
    
    pt_df["parcellation_index"] = pd.to_numeric(pt_df["parcellation_index"], errors="coerce")
    missing_mask = pt_df["parcellation_index"].isna()

    
    if missing_mask.any():
        pt_df.loc[missing_mask, "parcellation_index"] = _register_annotation_values(pt_df, parcellation_df)

    pt_df["parcellation_index"] = pt_df["parcellation_index"].astype(int)

    def _dedupe_by_parcellation_index(df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate rows that share the same parcellation_index."""
        with_index = df.dropna(subset=["parcellation_index"]).copy()
        without_index = df[df["parcellation_index"].isna()].copy()

        def _merge_group(g: pd.DataFrame) -> pd.Series:
            merged_term_sets = sorted(
                set(
                    g["term_set_name"]
                    .dropna()
                    .astype(str)
                    .loc[lambda s: s.str.strip() != ""]
                    .tolist()
                )
            )
            # Choose the canonical row.
            mask = g["label"].astype(str).str.startswith("AllenCCF", na=False)
            row = g.loc[mask].iloc[0].copy()
            row["term_set_name"] = merged_term_sets
            return row

        rows = []
        for _, g in with_index.groupby("parcellation_index", sort=False, dropna=False):
            if len(g) == 1:
                row = g.iloc[0].copy()
                val = row.get("term_set_name")
                # Normalize into a list-of-strings column (for Parquet friendliness)
                if not isinstance(val, list):
                    if pd.isna(val):
                        row["term_set_name"] = []
                    elif isinstance(val, str):
                        row["term_set_name"] = [val]
                rows.append(row)
            else:
                rows.append(_merge_group(g))

        deduped = pd.DataFrame(rows)
        return pd.concat([deduped, without_index], ignore_index=True)

    pt_df = _dedupe_by_parcellation_index(pt_df)

    # Confirm the column is consistently list-typed
    assert pt_df["term_set_name"].apply(lambda x: isinstance(x, list)).all()

    # Build DataFrame expected by Terminology (include term_set_name)
    filtered_df = pd.DataFrame(
        {
            "identifier": pt_df["identifier"],  # preserve NaN
            "parent_identifier": pt_df["parent_identifier"].map(lambda x: str(x) if not pd.isna(x) else ""),
            "name": pt_df["name"],
            "color_hex_triplet": pt_df["color_hex_triplet"],
            "abbreviation": pt_df["acronym"],
            "term_set_name": pt_df["term_set_name"],
            "annotation_value": pt_df["parcellation_index"].apply(lambda v: int(v)),
        }
    )

    return filtered_df


def create_ccf2020_terminology(input_dir, output_dir, library):
    """Create parcellation terminology from CCF 2020 metadata."""
    metadata_dir = Path(input_dir) / "metadata" / "Allen-CCF-2020" / "20230630"

    filtered_df = _build_ccf2020_terminology_dataframe(metadata_dir)

    pt = Terminology(
        df=filtered_df,
        name="allen-adult-mouse-terminology",
        version="2020",
    )

    # Compute descendant_annotation_values using descendant_identifiers
    id_to_ann = {
        ident: (vals if isinstance(vals, list) else ([vals] if pd.notna(vals) else []))
        for ident, vals in zip(pt.df["identifier"], pt.df["annotation_value"])
    }
    pt.df["descendant_annotation_values"] = pt.df["descendant_identifiers"].apply(
        lambda ids: sorted({x for ident in ids for x in (id_to_ann.get(ident) or [])})
    )

    # Copy all metadata files to the terminology directory
    parcellation_legacy_dir = pt.location(output_dir) / "legacy_files"

    for input_path in metadata_dir.glob("*.csv"):
        output_path = parcellation_legacy_dir / input_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)

    pt.write_terminology(output_dir)
    pt.create_manifest(output_dir)
    library.add(pt)

    # Write data description for terminology
    _write_ccf2020_terminology_data_description(pt.location(output_dir))

    return pt




def _write_ccf2020_annotation_data_description(output_dir: Path):
    """Write data_description.json for the 2020 stereotaxic annotation set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name(
            "allen-adult-mouse-stereotaxic-annotation-2020",
            CCF2020_ANNOTATION_CREATION_TIME,
        ),
        data_summary=CCF2020_ANNOTATION_DESCRIPTION.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],
        data_level="derived",
        creation_time=CCF2020_ANNOTATION_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Quanxin Wang", registry_identifier="0000-0002-0007-7935")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for 2020 annotation to {output_dir}")


def _write_ccf2020_terminology_data_description(output_dir: Path):
    """Write data_description.json for the 2020 terminology (ontology)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name("allen-adult-mouse-terminology-2020", CCF2020_TERMINOLOGY_CREATION_TIME),
        data_summary=CCF2020_TERMINOLOGY_DESCRIPTION.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],  # Derived from STPT population data & multimodal sources
        data_level="derived",
        creation_time=CCF2020_TERMINOLOGY_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Quanxin Wang", registry_identifier="0000-0002-0007-7935")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for 2020 terminology to {output_dir}")




def _write_ccf2020_template_data_description(output_dir: Path):
    """Write data_description.json for the 2020 anatomical template."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name("allen-adult-mouse-stpt-template-2020", CCF2020_TEMPLATE_CREATION_TIME),
        data_summary=CCF2020_TEMPLATE_DESCRIPTION.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],  # Derived from STPT imaging
        data_level="derived",
        creation_time=CCF2020_TEMPLATE_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Lydia Ng", registry_identifier="0000-0002-7499-3514")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for 2020 template to {output_dir}")


def create_ccf2020_annotation_set(input_dir, results_dir, library, scales=(10, 25)):
    """Create CCF 2020 anatomical annotation set with updated brain region labels."""
    logging.info("Creating CCF 2020 anatomical annotation set...")

    # Get required assets from library
    template = library.get_template("allen-adult-mouse-stpt-template", "2020")
    terminology = library.get_terminology(
        "allen-adult-mouse-terminology", "2020"
    )

    annotation_set = AnnotationSet(
        name="allen-adult-mouse-stereotaxic-annotation",
        coordinate_space=template.coordinate_space,
        template=template,
        terminology=terminology,
        version="2020",
        scales=scales,
    )

    annotation_dir = input_dir / "image_volumes" / "Allen-CCF-2020" / "20250331"
    annotation_set.create_from_nifti(
        input_prefix=annotation_dir / "annotation",
        output_root=results_dir,
        include_meshes=False,
    )

    annotation_output_dir = annotation_set.location(results_dir)
    uncompress_annotations_to_zarr(
        input_dir=annotation_output_dir,
        terminology=terminology,
        output_dir=annotation_output_dir,
        scales=annotation_set.scales,
    )

    # Write data description for 2020 annotation
    _write_ccf2020_annotation_data_description(annotation_set.location(results_dir))

    annotation_set.create_manifest(results_dir)

    def map_obj_id_to_annotation_value(obj_id):
        """Map object IDs to file IDs."""
        # Use the file_id from the terminology DataFrame
        val = terminology.df.loc[terminology.df["identifier"] == f"MBA:{obj_id}", "annotation_value"].values[0]
        return val[0] if isinstance(val, list) else val

    meshes = load_ccf3_meshes(Path("/data/ccf_meshes/mcc/annotation/ccf_2017/structure_meshes"))

    precomputed_dir = (
        results_dir
        / "annotation-sets"
        / "allen-adult-mouse-stereotaxic-annotation"
        / "2020"
        / "annotations.precomputed"
    )

    # Remove parent mesh fragments but keep mesh fragments from structures with voxels directly in the annotation set.
    parcellation_df = pd.read_csv(input_dir / "metadata" / "Allen-CCF-2020" / "20230630" / "parcellation.csv")
    clear_meshes_from_precomputed(
        precomputed_dir,
        keep_annotation_values=set(terminology.df["annotation_value"]) & set(parcellation_df["parcellation_index"]),
    )

    # Append meshes
    append_meshes_to_precomputed(
        ((m, map_obj_id_to_annotation_value(obj_id)) for m, obj_id in meshes),
        precomputed_dir,
        scale=1000,  # convert microns to nm
        map_annotation_value=lambda v: v[0] if isinstance(v, list) else v,
    )

    # Add to asset library
    library.add(annotation_set)

    logging.info("CCF 2020 anatomical annotation set created successfully")


def package_ccf2020(input_dir, output_dir, library, scales=(10,)):
    """Complete packaging workflow for CCF 2020 atlas data."""
    # Create and register anatomical template
    create_ccf2020_template(input_dir, output_dir, library, scales)

    # Create and register terminology
    create_ccf2020_terminology(input_dir, output_dir, library)

    # Create and register coordinate space
    coordinate_space = CoordinateSpace(
        name="allen-adult-mouse-ccf-stereotaxic-space",
        version="2020",
    )
    coordinate_space.create_manifest(output_dir)
    library.add(coordinate_space)

    library.get_template("allen-adult-mouse-stpt-template", "2020").coordinate_space = coordinate_space

    # Create and register annotation set
    create_ccf2020_annotation_set(input_dir, output_dir, library, scales)

    # Create parcellation atlas
    atlas = Atlas(
        name="allen-adult-mouse-ccf-stereotaxic-atlas",
        version="2020",
        coordinate_space=library.get_coordinate_space(
            "allen-adult-mouse-ccf-stereotaxic-space", "2020"
        ),
        templates=[library.get_template(
            "allen-adult-mouse-stpt-template", "2020"
        )],
        annotation_sets=[library.get_annotation_set(
            "allen-adult-mouse-stereotaxic-annotation", "2020"
        )],
    )
    atlas.create_manifest(output_dir)
    library.add(atlas)
