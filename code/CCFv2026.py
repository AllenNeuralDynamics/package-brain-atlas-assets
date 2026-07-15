"""Allen CCF 2026 annotation packaging with midline masks and hemisphere terms."""


import argparse
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, cast

import datetime
import shutil

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import zarr
from brainglobe_atlasapi import BrainGlobeAtlas 

import CCFv2020
from CCFv3 import load_ccf3_meshes  # type: ignore[import-not-found]
from atlas_builder import AnnotationSet, AssetLibrary, CoordinateSpace, Terminology  # type: ignore[import-not-found]
from atlas_builder.annotation_set import uncompress_annotations_to_zarr  # type: ignore[import-not-found]
from atlas_builder.precomputed import append_meshes_to_precomputed  # type: ignore[import-not-found]


from aind_data_schema.core.data_description import DataDescription, Funding  # type: ignore[import-not-found]
from aind_data_schema_models.data_name_patterns import build_data_name  # type: ignore[import-not-found]
from aind_data_schema_models.modalities import Modality  # type: ignore[import-not-found]
from aind_data_schema_models.organizations import Organization  # type: ignore[import-not-found]
from aind_data_schema.components.identifiers import Person  # type: ignore[import-not-found]

from CCFv2020 import _build_ccf2020_terminology_dataframe  # type: ignore[import-not-found]


BRAINGLOBE_DATA_DIR = Path("/data/.brainglobe")
DEFAULT_CCF2026_SCALES = (10, 25)

CCF2026_TERMINOLOGY_DESCRIPTION = (
    "The 2026-03 revision of the Allen Mouse Reference Atlas, Ontology matches the 2020 "
    "release, with four additional rows for hemispheric labels (Left of midline, Right of "
    "midline, Left hemisphere, and Right hemisphere)."
)

CCF2026_TERMINOLOGY_CREATION_TIME = datetime.datetime(2026, 3, 13, tzinfo=datetime.timezone.utc)
CCF2026_ANNOTATION_CREATION_TIME = datetime.datetime(2026, 3, 13, tzinfo=datetime.timezone.utc)

CCF2026_ANNOTATION_DESCRIPTION = (
    "The 2026-03 revision of the Allen Mouse Common Coordinate Framework annotation matches the "
    "2020 release, with four additional annotations for left and right of midline and the left and right hemispheres. "
    "No changes were made to the compressed annotation values or meshes."
)


def _write_ccf2026_terminology_data_description(output_dir: Path):
    """Write data_description.json for the 2026 terminology (ontology)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name("allen-adult-mouse-terminology-2026-03", CCF2026_TERMINOLOGY_CREATION_TIME),
        data_summary=CCF2026_TERMINOLOGY_DESCRIPTION.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],  # Derived from STPT population data & multimodal sources
        data_level="derived",
        creation_time=CCF2026_TERMINOLOGY_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Quanxin Wang", registry_identifier="0000-0002-0007-7935")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for 2026 terminology to {output_dir}")


def _write_ccf2026_annotation_data_description(output_dir: Path):
    """Write data_description.json for the 2026-03 stereotaxic annotation set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDescription(
        name=build_data_name(
            "allen-adult-mouse-stereotaxic-annotation-2026-03",
            CCF2026_ANNOTATION_CREATION_TIME,
        ),
        data_summary=CCF2026_ANNOTATION_DESCRIPTION.strip(),
        subject_id="adult-mouse-population-average",
        modalities=[Modality.STPT],
        data_level="derived",
        creation_time=CCF2026_ANNOTATION_CREATION_TIME,
        institution=Organization.AIBS,
        investigators=[Person(name="Quanxin Wang", registry_identifier="0000-0002-0007-7935")],
        funding_source=[Funding(funder=Organization.AI)],
        project_name="Allen Mouse Brain Common Coordinate Framework",
    )
    dd.write_standard_file(output_directory=output_dir)
    logging.info(f"Wrote data_description.json for 2026 annotation to {output_dir}")


def create_ccf2026_terminology(input_dir, output_dir, library):
    """Create the 2026-03 revision of the CCF 2020 terminology with midline and hemisphere rows."""
    metadata_dir = Path(input_dir) / "metadata" / "Allen-CCF-2020" / "20230630"

    filtered_df = _build_ccf2020_terminology_dataframe(metadata_dir)

    max_ann = pd.to_numeric(filtered_df["annotation_value"], errors="coerce").max()
    max_ann = int(max_ann) if pd.notna(max_ann) else 0

    row_template = {col: pd.NA for col in filtered_df.columns}
    row_template.update(
        {
            "term_set_name": [],
        }
    )

    hemisphere_rows = [
        {
            **row_template,
            "identifier": f"MBA:{max_ann + 1}",
            "parent_identifier": "",
            "name": "Left of midline",
            "abbreviation": "LMid",
            "annotation_value": max_ann + 1,
            "color_hex_triplet": "#666666",
        },
        {
            **row_template,
            "identifier": f"MBA:{max_ann + 2}",
            "parent_identifier": "",
            "name": "Right of midline",
            "abbreviation": "RMid",
            "annotation_value": max_ann + 2,
            "color_hex_triplet": "#888888",
        },
        {
            **row_template,
            "identifier": f"MBA:{max_ann + 3}",
            "parent_identifier": "MBA:997",
            "name": "Left hemisphere",
            "abbreviation": "LHem",
            "annotation_value": max_ann + 3,
            "color_hex_triplet": "#666666",
        },
        {
            **row_template,
            "identifier": f"MBA:{max_ann + 4}",
            "parent_identifier": "MBA:997",
            "name": "Right hemisphere",
            "abbreviation": "RHem",
            "annotation_value": max_ann + 4,
            "color_hex_triplet": "#888888",
        },
    ]

    filtered_df = pd.concat([filtered_df, pd.DataFrame(hemisphere_rows)], ignore_index=True)

    terminology = Terminology(
        df=filtered_df,
        name="allen-adult-mouse-terminology",
        version="2026-03",
    )

    id_to_ann = {
        ident: (vals if isinstance(vals, list) else ([vals] if pd.notna(vals) else []))
        for ident, vals in zip(terminology.df["identifier"], terminology.df["annotation_value"])
    }
    terminology.df["descendant_annotation_values"] = terminology.df["descendant_identifiers"].apply(
        lambda ids: sorted({x for ident in ids for x in (id_to_ann.get(ident) or [])})
    )

    parcellation_legacy_dir = terminology.location(output_dir) / "legacy_files"

    for input_path in metadata_dir.glob("*.csv"):
        output_path = parcellation_legacy_dir / input_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)

    terminology.write_terminology(output_dir)
    terminology.create_manifest(output_dir)
    library.add(terminology)

    _write_ccf2026_terminology_data_description(terminology.location(output_dir))

    return terminology


class TerminologyLike(Protocol):
    """Protocol for terminology objects with a DataFrame."""

    df: "pd.DataFrame"


def _load_brainglobe_atlas(atlas_name: str, data_dir: Path) -> BrainGlobeAtlas:
    """Load a BrainGlobe atlas, preferring the provided data directory."""
    atlas_dir = Path(data_dir)
    if not atlas_dir.exists():
        raise FileNotFoundError(f"BrainGlobe data directory does not exist: {atlas_dir}")

    os.environ["BRAINGLOBE_CONFIG_DIR"] = str(atlas_dir)

    return BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=atlas_dir,
        interm_download_dir=atlas_dir,
        check_latest=False,
    )


def _extract_hemispheres(atlas: object) -> np.ndarray:
    """Extract hemisphere labels from a BrainGlobe atlas instance."""
    for attr in ("hemispheres", "hemisphere"):
        if hasattr(atlas, attr):
            value = cast(object, getattr(atlas, attr))
            if callable(value):
                data = cast(Callable[[], object], value)()
            else:
                data = value
            return np.asarray(data)
    if hasattr(atlas, "get_hemispheres"):
        getter = cast(Callable[[], object], getattr(atlas, "get_hemispheres"))
        data = getter()
        return np.asarray(data)
    raise ValueError("BrainGlobe atlas does not expose hemisphere labels")


def _midline_masks_from_brainglobe(resolution: int, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load left/right midline-based masks for a specific resolution."""
    atlas_name = f"allen_mouse_{resolution}um"
    atlas = _load_brainglobe_atlas(atlas_name, data_dir)
    root_mask = atlas.annotation > 0

    hemispheres = np.asarray(_extract_hemispheres(atlas))
    values = set(np.unique(hemispheres).tolist())

    if {1, 2}.issubset(values):
        left_mask = hemispheres == 1
        right_mask = hemispheres == 2
    else:
        raise ValueError(f"Unexpected midline label values {sorted(values)} for atlas {atlas_name}.")

    return left_mask.astype(np.uint8), right_mask.astype(np.uint8), root_mask.astype(np.uint8)


def _get_bilateral_annotation_values(terminology: TerminologyLike) -> tuple[int, int, int, int]:
    """Return annotation values for left/right of midline from terminology."""
    df = terminology.df

    def _find_value(name: str, abbreviation: str) -> int:
        name_mask = df["name"].astype(str).str.lower() == name.lower()
        if name_mask.any():
            return int(df.loc[name_mask, "annotation_value"].iloc[0])

        abbr_mask = df["abbreviation"].astype(str).str.lower() == abbreviation.lower()
        if abbr_mask.any():
            return int(df.loc[abbr_mask, "annotation_value"].iloc[0])

        raise ValueError(f"Could not locate midline term '{name}' in terminology")

    lmid_value = _find_value("Left of midline", "LMid")
    rmid_value = _find_value("Right of midline", "RMid")
    lh_value = _find_value("Left hemisphere", "LHem")
    rh_value = _find_value("Right hemisphere", "RHem")

    return lmid_value, rmid_value, lh_value, rh_value


def _scale_to_dataset_index(
    annotation_output_dir: Path,
    scales: Iterable[int],
    annotations_grp: zarr.Group,
) -> dict[int, str]:
    """Map each scale to its dataset index in annotations.ome.zarr."""
    scale_to_index = {}
    scale_index = 0
    for scale in scales:
        src_fname = annotation_output_dir / f"annotations_compressed_{scale}.nii.gz"
        if not src_fname.exists():
            continue
        dataset_name = f"s{scale_index}"
        if dataset_name in annotations_grp:
            scale_to_index[scale] = dataset_name
        scale_index += 1
    return scale_to_index


def _add_midline_masks_to_uncompressed(
    annotation_output_dir: Path,
    terminology,
    scales,
    brainglobe_dir: Path,
):
    """Add left/right midline-based masks into the uncompressed OME-Zarr."""
    zarr_path = annotation_output_dir / "annotations.ome.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Missing uncompressed annotations at {zarr_path}")

    group = zarr.open(str(zarr_path), mode="r+")
    annotations_grp = group

    lmid_value, rmid_value, lhem_value, rhem_value = _get_bilateral_annotation_values(terminology)
    scale_to_index: dict[int, str] = _scale_to_dataset_index(
        annotation_output_dir, scales, annotations_grp
    )

    annotation_values_ds = cast(zarr.Array, annotations_grp["annotation_values"])
    annotation_values = np.asarray(annotation_values_ds[:])

    def _append_annotation_value(value: int):
        nonlocal annotation_values
        if value in annotation_values:
            return
        new_len = len(annotation_values) + 1
        _ = annotation_values_ds.resize((new_len,))
        annotation_values_ds[new_len - 1] = value
        annotation_values = np.append(annotation_values, value)

        for dataset_name in scale_to_index.values():
            zarr_array = cast(zarr.Array, annotations_grp[dataset_name])
            new_shape = (new_len,) + zarr_array.shape[1:]
            _ = zarr_array.resize(new_shape)
            zarr_array[new_len - 1, :, :, :] = 0

    _append_annotation_value(lmid_value)
    _append_annotation_value(rmid_value)
    _append_annotation_value(lhem_value)
    _append_annotation_value(rhem_value)

    lmid_idx = np.where(annotation_values == lmid_value)[0]
    rmid_idx = np.where(annotation_values == rmid_value)[0]
    lhem_idx = np.where(annotation_values == lhem_value)[0]
    rhem_idx = np.where(annotation_values == rhem_value)[0]

    if lmid_idx.size != 1 or rmid_idx.size != 1:
        raise ValueError(
            "Midline annotation values not found uniquely in annotation_values"
        )

    for scale, dataset_name in scale_to_index.items():
        lmid_mask, rmid_mask, root_mask = _midline_masks_from_brainglobe(scale, brainglobe_dir)

        zarr_array = annotations_grp[dataset_name]
        if zarr_array.shape[1:] != lmid_mask.shape:
            raise ValueError(
                f"Midline mask shape {lmid_mask.shape} does not match "
                f"annotation shape {zarr_array.shape[1:]} for scale {scale}"
            )

        logging.info(f"Writing midline masks for scale {scale} into dataset {dataset_name}")
        zarr_array[lmid_idx[0], :, :, :] = lmid_mask
        zarr_array[rmid_idx[0], :, :, :] = rmid_mask
        zarr_array[lhem_idx[0], :, :, :] = root_mask & lmid_mask
        zarr_array[rhem_idx[0], :, :, :] = root_mask & rmid_mask


def create_ccf2026_annotation_set(
    input_dir,
    results_dir,
    library,
    scales=DEFAULT_CCF2026_SCALES,
    brainglobe_dir: Path = BRAINGLOBE_DATA_DIR,
):
    """Create the 2026 CCF annotation set and add midline masks."""
    logging.info("Creating CCF 2026 anatomical annotation set...")

    template = library.get_template("allen-adult-mouse-stpt-template", "2020")
    terminology = library.get_terminology("allen-adult-mouse-terminology", "2026-03")

    coordinate_space = template.coordinate_space
    if coordinate_space is None:
        coordinate_space = CoordinateSpace(
            name="allen-adult-mouse-ccf-stereotaxic-space",
            version="2020",
        )
        coordinate_space.create_manifest(results_dir)
        library.add(coordinate_space)
        template.coordinate_space = coordinate_space

    annotation_set = AnnotationSet(
        name="allen-adult-mouse-stereotaxic-annotation",
        coordinate_space=coordinate_space,
        template=template,
        terminology=terminology,
        version="2026-03",
        scales=scales,
    )

    annotation_dir = Path(input_dir) / "image_volumes" / "Allen-CCF-2020" / "20250331"
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

    _write_ccf2026_annotation_data_description(annotation_set.location(results_dir))

    _add_midline_masks_to_uncompressed(
        annotation_output_dir=annotation_output_dir,
        terminology=terminology,
        scales=annotation_set.scales,
        brainglobe_dir=brainglobe_dir,
    )

    annotation_set.create_manifest(results_dir)

    def map_obj_id_to_annotation_value(obj_id):
        """Map object IDs to file IDs."""
        val = terminology.df.loc[
            terminology.df["identifier"] == f"MBA:{obj_id}", "annotation_value"
        ].values[0]
        return val[0] if isinstance(val, list) else val

    meshes = load_ccf3_meshes(Path("/data/ccf_meshes/mcc/annotation/ccf_2017/structure_meshes"))

    append_meshes_to_precomputed(
        ((m, map_obj_id_to_annotation_value(obj_id)) for m, obj_id in meshes),
        results_dir
        / "annotation-sets"
        / "allen-adult-mouse-stereotaxic-annotation"
        / "2026-03"
        / "annotations.precomputed",
        scale=1000,
        map_annotation_value=lambda v: v[0] if isinstance(v, list) else v,
    )

    library.add(annotation_set)
    logging.info("CCF 2026 anatomical annotation set created successfully")


def package_ccf2026(
    input_dir: str | Path,
    output_dir: str | Path,
    library: object,
    scales: tuple[int, ...] = DEFAULT_CCF2026_SCALES,
):
    """Package the 2026-03 terminology and annotation set."""
    _ = create_ccf2026_terminology(input_dir, output_dir, library)
    create_ccf2026_annotation_set(input_dir, output_dir, library, scales=scales)


def run_ccf2026_standalone(
    input_dir: str | Path,
    output_dir: str | Path,
    scales: tuple[int, ...] = DEFAULT_CCF2026_SCALES,
    brainglobe_dir: str | Path = BRAINGLOBE_DATA_DIR,
    include_annotation: bool = True,
    include_terminology: bool = True,
) -> AssetLibrary:
    """Run the minimal dependency chain required to package CCF 2026 on its own."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    brainglobe_path = Path(brainglobe_dir)

    logging.info("Running standalone CCF 2026 packaging")
    library = AssetLibrary()

    CCFv2020.create_ccf2020_template(input_path, output_path, library, scales=scales)

    if include_terminology:
        create_ccf2026_terminology(input_path, output_path, library)

    if include_annotation:
        if not include_terminology:
            _ = create_ccf2026_terminology(input_path, output_path, library)
        create_ccf2026_annotation_set(
            input_path,
            output_path,
            library,
            scales=scales,
            brainglobe_dir=brainglobe_path,
        )

    return library


def _parse_scales(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integer scales."""
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Scales must be a comma-separated list of integers") from exc


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone CCF 2026 packaging."""
    parser = argparse.ArgumentParser(description="Package the CCF 2026 atlas assets only.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/abc_atlas"),
        help="ABC Atlas input directory (default: /data/abc_atlas)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("/results"),
        help="Directory to write CCF 2026 outputs (default: /results)",
    )
    parser.add_argument(
        "--brainglobe-dir",
        type=Path,
        default=BRAINGLOBE_DATA_DIR,
        help="BrainGlobe atlas cache directory (default: /data/.brainglobe)",
    )
    parser.add_argument(
        "--scales",
        type=_parse_scales,
        default=DEFAULT_CCF2026_SCALES,
        help="Comma-separated voxel scales to package (default: 10,25)",
    )
    parser.add_argument(
        "--terminology-only",
        action="store_true",
        help="Only build the 2026 terminology and skip the annotation set.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for standalone CCF 2026 packaging."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _ = run_ccf2026_standalone(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scales=args.scales,
        brainglobe_dir=args.brainglobe_dir,
        include_annotation=not args.terminology_only,
        include_terminology=True,
    )


if __name__ == "__main__":
    main()
