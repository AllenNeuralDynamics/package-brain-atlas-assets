"""Allen CCF 2020 atlas packaging from ABC Atlas data."""

import logging
import shutil
from pathlib import Path

import pandas as pd
from CCFv3 import load_ccf3_meshes

from atlas_assets import (AnatomicalAnnotationSet, AnatomicalSpace,
                          AnatomicalTemplate, ParcellationAtlas,
                          ParcellationTerminology)
from atlas_assets.precomputed import append_meshes_to_precomputed


def create_ccf2020_anatomical_template(input_dir, results_dir, library, scales=(10,)):
    """Create CCF 2020 anatomical template from ABC Atlas data."""
    logging.info("Creating CCF 2020 anatomical template...")

    # Create anatomical template from the ABC Atlas average template
    template_dir = input_dir / "image_volumes" / "Allen-CCF-2020" / "20230630"
    template = AnatomicalTemplate(
        name="allen-adult-mouse-2p-template", version="2020", scales=scales
    )

    # Use the average template from ABC Atlas
    template.create(template_dir / "average_template", results_dir)
    library.add(template)
    logging.info(f"Created CCF 2020 template: {template.name} {template.version}")

    return template


def create_ccf2020_parcellation_terminology(input_dir, output_dir, library):
    """Create parcellation terminology from CCF 2020 metadata."""
    metadata_dir = Path(input_dir) / "metadata" / "Allen-CCF-2020" / "20230630"

    # Load inputs
    pt_df = pd.read_csv(metadata_dir / "parcellation_term.csv")
    pptm_df = pd.read_csv(
        metadata_dir / "parcellation_to_parcellation_term_membership.csv"
    )

    # 1) Filter membership for 'substructure' term set
    substructure_membership = pptm_df[
        pptm_df["parcellation_term_set_name"] == "substructure"
    ].copy()

    # 2) Map each parcellation_term_label to parcellation_index (as list)
    label_to_indices = {}
    for label in substructure_membership["parcellation_term_label"].unique():
        idxs = sorted(
            substructure_membership[
                substructure_membership["parcellation_term_label"] == label
            ]["parcellation_index"].unique()
        )
        label_to_indices[label] = [int(i) for i in idxs]

    # 3) Populate annotation_value for each term (no minting yet)
    pt_df = pt_df.copy()
    pt_df["annotation_value"] = pt_df["label"].map(label_to_indices)

    # 3a) Prepare term_set_name mapping from a slimmed membership (drop parcellation columns, dedupe)
    membership_slim = pptm_df.drop(
        columns=["parcellation_index", "voxel_count", "volume_mm3"], errors="ignore"
    ).drop_duplicates()
    membership_slim_sorted = membership_slim.sort_values(
        ["parcellation_term_label", "term_set_order"], ascending=[True, False]
    )
    preferred_term_set = membership_slim_sorted.drop_duplicates(
        subset=["parcellation_term_label"], keep="first"
    )
    label_to_term_set = dict(
        zip(
            preferred_term_set["parcellation_term_label"],
            preferred_term_set["parcellation_term_set_name"],
        )
    )

    # 4) Build groups DataFrame, add term_set_name BEFORE dropping 'label'
    df_group = pt_df.copy()
    df_group["term_set_name"] = df_group["label"].map(label_to_term_set)
    df_group = df_group.drop(columns=["label"]).copy()

    # Split rows by identifier presence
    nonnull = df_group[df_group["identifier"].notna()].copy()
    missing_id_rows = df_group[df_group["identifier"].isna()].copy()

    # Representative row (first) for other columns per identifier
    rep = nonnull.groupby("identifier", as_index=False).first()

    # Flatten, deduplicate, sort annotation values per identifier
    agg_ann = nonnull.groupby("identifier", as_index=False)["annotation_value"].agg(
        lambda s: sorted({x for lst in s if isinstance(lst, list) for x in lst})
    )

    if "annotation_value" in rep.columns:
        rep = rep.drop(columns=["annotation_value"])
    grouped = rep.merge(agg_ann, on="identifier", how="left")

    # Ensure missing-identifier rows have deduped/sorted annotation_value
    if not missing_id_rows.empty:
        missing_id_rows["annotation_value"] = missing_id_rows["annotation_value"].apply(
            lambda lst: sorted(set(lst)) if isinstance(lst, list) else lst
        )

    combined = pd.concat([grouped, missing_id_rows], ignore_index=True, sort=False)

    # 5) Mint new IDs after deduplication/grouping
    all_existing = []
    for v in combined["annotation_value"].dropna():
        if isinstance(v, list):
            all_existing.extend(v)
    # include membership indices as well
    all_existing.extend(
        int(x) for x in pptm_df["parcellation_index"].unique() if pd.notna(x)
    )
    max_existing = max(all_existing) if all_existing else 0
    next_new_id = max_existing + 1

    # Assign new IDs to rows with missing/empty annotation_value
    need_ids_mask = combined["annotation_value"].isna() | combined[
        "annotation_value"
    ].apply(lambda v: isinstance(v, list) and len(v) == 0)
    for i in combined[need_ids_mask].index:
        combined.at[i, "annotation_value"] = [next_new_id]
        next_new_id += 1

    # Build DataFrame expected by ParcellationTerminology (include term_set_name)
    filtered_df = pd.DataFrame(
        {
            "identifier": combined["identifier"],  # preserve NaN
            "parent_identifier": combined["parent_identifier"].map(
                lambda x: str(x) if not pd.isna(x) else ""
            ),
            "name": combined["name"],
            "color_hex_triplet": combined["color_hex_triplet"],
            "abbreviation": combined["acronym"],
            "term_set_name": combined["term_set_name"],
            "annotation_value": combined["annotation_value"],
        }
    )

    pt = ParcellationTerminology(
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

    return pt


def create_ccf2020_annotation_set(input_dir, results_dir, library, scales=(10,)):
    """Create CCF 2020 anatomical annotation set with updated brain region labels."""
    logging.info("Creating CCF 2020 anatomical annotation set...")

    # Get required assets from library
    template = library.get_anatomical_template("allen-adult-mouse-2p-template", "2020")
    terminology = library.get_parcellation_terminology(
        "allen-adult-mouse-terminology", "2020"
    )

    annotation_set = AnatomicalAnnotationSet(
        name="allen-adult-mouse-stereotaxic-annotation",
        anatomical_template=template,
        parcellation_terminology=terminology,
        version="2020",
        scales=scales,
    )

    annotation_dir = input_dir / "image_volumes" / "Allen-CCF-2020" / "20250331"
    annotation_set.create_from_nifti(
        input_prefix=annotation_dir / "annotation", output_root=results_dir
    )

    annotation_set.create_manifest(results_dir)

    def map_obj_id_to_annotation_value(obj_id):
        """Map object IDs to file IDs."""
        # Use the file_id from the terminology DataFrame
        val = terminology.df.loc[
            terminology.df["identifier"] == f"MBA:{obj_id}", "annotation_value"
        ].values[0]
        return val[0] if isinstance(val, list) else val

    meshes = load_ccf3_meshes(
        Path("/data/ccf_meshes/mcc/annotation/ccf_2017/structure_meshes")
    )

    # Append just meshes; segment properties already written by create()
    append_meshes_to_precomputed(
        ((m, map_obj_id_to_annotation_value(obj_id)) for m, obj_id in meshes),
        results_dir
        / "anatomical-annotation-sets"
        / "allen-adult-mouse-stereotaxic-annotation"
        / "2020"
        / "annotations.precomputed",
        scale=1000,  # convert microns to nm
        map_annotation_value=lambda v: v[0] if isinstance(v, list) else v,
    )

    # Add to asset library
    library.add(annotation_set)

    logging.info("CCF 2020 anatomical annotation set created successfully")


def package_ccf2020(input_dir, output_dir, library, scales=(10,)):
    """Complete packaging workflow for CCF 2020 atlas data."""
    # Create and register anatomical template
    create_ccf2020_anatomical_template(input_dir, output_dir, library, scales)

    # Create and register terminology
    create_ccf2020_parcellation_terminology(input_dir, output_dir, library)

    # Create and register annotation set
    create_ccf2020_annotation_set(input_dir, output_dir, library, scales)

    # Create and register anatomical space
    template = library.get_anatomical_template("allen-adult-mouse-2p-template", "2020")
    anatomical_space = AnatomicalSpace(
        name="allen-adult-mouse-ccf-stereotaxic-space",
        version="2020",
        anatomical_template=template,
    )
    anatomical_space.create_manifest(output_dir)
    library.add(anatomical_space)

    # Create parcellation atlas
    atlas = ParcellationAtlas(
        name="allen-adult-mouse-ccf-stereotaxic-atlas",
        version="2020",
        anatomical_space=library.get_anatomical_space(
            "allen-adult-mouse-ccf-stereotaxic-space", "2020"
        ),
        anatomical_annotation_set=library.get_anatomical_annotation_set(
            "allen-adult-mouse-stereotaxic-annotation", "2020"
        ),
        parcellation_terminology=library.get_parcellation_terminology(
            "allen-adult-mouse-terminology", "2020"
        ),
    )
    atlas.create_manifest(output_dir)
    library.add(atlas)
