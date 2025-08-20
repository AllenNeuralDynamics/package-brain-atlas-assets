import json
import logging
from pathlib import Path

import cloudvolume
import numpy as np
import tensorstore as ts
from zmesh import Mesher


def append_mesh_to_precomputed(mesh, scale, precomputed_file, identifier):
    cvmesh = cloudvolume.Mesh(
        mesh.vertices[:, [2, 1, 0]] * scale,
        mesh.face_vertices,
        normals=mesh.vertex_normals,
    )

    cvbytes = cvmesh.to_precomputed()

    (Path(precomputed_file) / "mesh").mkdir(parents=True, exist_ok=True)
    cv_filename = f"{precomputed_file}/mesh/{identifier}:0:0"
    cv_manifest = f"{precomputed_file}/mesh/{identifier}:0"
    manifest = {"fragments": [f"{identifier}:0:0"]}

    # write the manifest
    with open(cv_manifest, "w") as f:
        json.dump(manifest, f)
    with open(cv_filename, "wb") as f:
        f.write(cvbytes)


def append_meshes_to_precomputed(meshes, precomputed_file, scale, map_annotation_value=None, mesh_dir="mesh"):
    """Append meshes to an existing precomputed dataset and update info (moved)."""
    map_annotation_value = map_annotation_value or (lambda x: x)

    for mesh, annotation_value in meshes:
        av = map_annotation_value(annotation_value)
        logging.info(f"Processing mesh id: {annotation_value}, mapped id: {av}")
        append_mesh_to_precomputed(mesh, scale, precomputed_file, av)

    # Update info to point to mesh dir
    info_path = Path(precomputed_file) / "info"
    if info_path.exists():
        with open(info_path, "r") as f:
            existing_info = json.load(f)
    else:
        existing_info = {}
    existing_info.update({"mesh": mesh_dir})
    with open(info_path, "w") as f:
        json.dump(existing_info, f, indent=2)


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
                term_set_values.append(row.get("term_set_name") if row.get("term_set_name") else None)

    properties = [
        {
            "id": "abbreviation",
            "type": "label",
            "values": abbrev_labels,
        }
    ]

    if has_term_set:
        unique_term_set_names = sorted({v for v in term_set_values if v})
        lut = {k: i for i, k in enumerate(unique_term_set_names)}
        properties.append(
            {
                "id": "term set",
                "type": "tags",
                "tags": unique_term_set_names,
                "values": [[lut[v]] if v in lut else [] for v in term_set_values],
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

def _compute_meshes_from_annotation(terminology_ids, 
                                    annotation_ids:set,
                                    annotation: np.ndarray,
                                    voxel_spacing: list,
                                    mesher):

    meshes = []
    for struct_id, descendants in terminology_ids:

        match_ids = set(descendants) & annotation_ids
        struct_mask = np.zeros_like(annotation)
        struct_mask[np.isin(annotation,match_ids)] = struct_id

        mesher.mesh(struct_mask, close=True)

        meshes.append(
            [
                mesher.get(
                    struct_id,
                    normals=True,  # whether to calculate normals or not
                    reduction_factor=2,
                    max_error=8,  # Maximum tolerable error in physical space (um)
                    voxel_centered=False,  # Voxel center set to index, eg. [0,0,0]
                ),
                struct_id,
            ]
        )
        mesher.erase(struct_id)  # delete high res mesh for memory``
        mesher.clear()
    
    return meshes

def create_mesh_from_annotation(
    annotation_set: dict,
    mesh_directory: Path,
    chunk_size=(256, 256, 64),
):
    """Create meshes from annotation volume using zmesh's Mesher function"""

    meshes_dir = mesh_directory / "mesh"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    scale = annotation_set.scale

    voxel_spacing = [1000 * min(scale)] * 3  # Assumes isometric voxels. Multiplied by 1000 to get units into nm
    
    # Get annotation and terminology from annotation_set (see AnatomicalAnnotationSet) and extract identifiers
    annotation, _ = annotation_set.compressed_results[min(scale)]
    terminology = annotation_set.terminology

    annotation_ids = set(np.unique(annotation)[1:]) # Leaving out 0 (background)
    terminology_ids = zip(terminology.df["identifier"], terminology.df["descendant_identifiers"])
    
    # Initialize mesher with voxel size in nm
    mesher = Mesher(voxel_spacing)
    meshes = _compute_meshes_from_annotation(terminology_ids, annotation_ids, annotation, voxel_spacing, mesher)

    for region_mesh in meshes:

        mesh = region_mesh[0]
        region_id = region_mesh[1]  # Same as struct_id above

        mesh = mesher.compute_normals(mesh)

        cv_manifest = meshes_dir / f"{region_id}:0"
        manifest_json = {"fragments": [f"{region_id}:0:0"]}
        # write the manifest
        with open(cv_manifest, "w") as f:
            json.dump(manifest_json, f)

        save_filename = meshes_dir / f"{region_id}:0:0"
        with open(save_filename, "wb") as f:
            f.write(mesh.to_precomputed())

    info_d = {
        "@type": "neuroglancer_legacy_mesh",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": chunk_size,
                "compressed_segmentation_block_size": [8, 8, 8],
                "encoding": "compressed_segmentation",
                "key": "_".join([str(x) for x in voxel_spacing]),
                "resolution": voxel_spacing,
                "size": list(annotation.shape),
            }
        ],
        "type": "segmentation",
        "mesh": "mesh",
        "segment_properties": "segment_properties",
    }

    # write to an info file in the ccf_meshes folder
    with open((meshes_dir / "info"), "w") as f:
        json.dump(info_d, f)