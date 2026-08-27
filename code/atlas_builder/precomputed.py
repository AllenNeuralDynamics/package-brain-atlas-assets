import json
import logging
from pathlib import Path

import numpy as np
import tensorstore as ts


def write_segment_properties(
    precomputed_file,
    terms,
    map_annotation_value=None,
    segment_properties_dir="segment_properties",
):
    """Write neuroglancer segment properties based on a terminology DataFrame (moved)."""
    map_annotation_value = map_annotation_value or (lambda x: x)

    # Create separate entries for each annotation value
    flat_ids = []
    abbrev_labels = []
    term_set_values = []

    has_term_set = "term_set_name" in terms.columns

    for idx, row in terms.iterrows():
        av = row.get("annotation_value")
        if isinstance(av, list):
            values = av
        elif av is None or (isinstance(av, float) and np.isnan(av)):
            values = []
        else:
            values = [av]

        # Create an entry for each annotation value in this row
        for v in values:
            flat_ids.append(v)
            abbr = row.get("abbreviation", "")
            name = row.get("name", "")
            abbrev_labels.append(f"{abbr}: {name}".strip(": "))
            if has_term_set:
                term_set_values.append(row.get("term_set_name") if row.get("term_set_name") else [])

    properties = [
        {
            "id": "abbreviation",
            "type": "label",
            "values": abbrev_labels,
        }
    ]

    if has_term_set:
        unique_term_set_names = sorted({v for v_list in term_set_values for v in v_list})
        lut = {k: i for i, k in enumerate(unique_term_set_names)}
        properties.append(
            {
                "id": "term set",
                "type": "tags",
                "tags": unique_term_set_names,
                "values": [[lut[v] for v in v_list] for v_list in term_set_values],
            }
        )

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [str(map_annotation_value(v)) for v in flat_ids],
            "properties": properties,
        },
    }

    seg_dir = Path(precomputed_file) / segment_properties_dir
    seg_dir.mkdir(parents=True, exist_ok=True)

    with open(seg_dir / "info", "w") as f:
        json.dump(segment_properties, f, indent=2)

    # Update precomputed info to reference segment_properties
    info_path = Path(precomputed_file) / "info"
    if info_path.exists():
        with open(info_path, "r") as f:
            existing_info = json.load(f)
    else:
        existing_info = {}
    existing_info.update({"segment_properties": segment_properties_dir})
    with open(info_path, "w") as f:
        json.dump(existing_info, f, indent=2)


def convert_compressed_annotations_to_precomputed(
    annotation_data,
    output_location,
    scale=(0.01, 0.01, 0.01),
    chunk_size=(256, 256, 64),
):
    """Convert compressed annotation data to precomputed format (moved)."""
    logging.info("Converting compressed annotations to precomputed format")
    logging.info(f"Data shape: {annotation_data.shape}, dtype: {annotation_data.dtype}")
    logging.info(f"Output location: {output_location}")

    output_location = Path(output_location).resolve()
    output_location.mkdir(parents=True, exist_ok=True)

    # zyx -> xyz
    annotation_data = annotation_data.T
    scale = scale[::-1]

    spec = {
        "kvstore": {
            "driver": "file",
            "path": str(output_location),
        },
        "driver": "neuroglancer_precomputed",
        "dtype": "uint32",
        "scale_metadata": {
            "size": annotation_data.shape,
            "encoding": "compressed_segmentation",
            "compressed_segmentation_block_size": [8, 8, 8],
            "chunk_size": chunk_size,
            "resolution": [
                scale[0] * 1e6,
                scale[1] * 1e6,
                scale[2] * 1e6,
            ],  # must convert to nanometers
        },
        "schema": {
            "domain": {
                "exclusive_max": [
                    annotation_data.shape[0],
                    annotation_data.shape[1],
                    annotation_data.shape[2],
                    1,
                ],
                "inclusive_min": [0, 0, 0, 0],
                "labels": ["x", "y", "z", "channel"],
            },
            "rank": 4,
        },
        "create": True,
        "delete_existing": True,
    }

    store = ts.open(spec).result()
    store[:, :, :, 0].write(annotation_data.astype(np.uint32)).result()

    logging.info(f"Successfully created precomputed annotation file at {output_location}")
