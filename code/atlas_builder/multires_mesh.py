"""Multi-resolution (neuroglancer_multilod_draco) mesh generation via igneous.

Meshes are generated from the precomputed segmentation volume rather than from an
in-memory annotation array, so the volume must already be written before calling
create_multires_meshes.

Two properties of the output are deliberate and load-bearing:

Hierarchy. The volume holds leaf labels only, but every structure in the terminology
needs a mesh covering the union of its descendants' voxels. igneous meshes whatever
labels are present, so each structure's descendants are remapped onto it before
marching cubes. A remap is a function on voxels, so one pass can only realize one
antichain of the tree -- hence one meshing pass per hierarchy depth. Structures at a
given depth have disjoint descendant sets, and each structure sits at exactly one
depth, so the per-pass label sets never collide and all passes can write into a
single mesh directory.

Single level of detail. igneous's unsharded multires path tears geometry whenever a
level of detail is split into more than one octree fragment: fragment vertices are
quantized onto per-fragment grids, so vertices on a shared cut plane stop coinciding
and the surface opens up. Watertight and multi-LOD are mutually exclusive there, and
watertight is what the atlas needs, so exactly one LOD is emitted.
"""

import logging
import re
from pathlib import Path

import numpy as np

# Pass-1 intermediates are named "{label}:{lod}:{bbox}"; the final multires output for
# a label is "{label}" plus "{label}.index".
_INTERMEDIATE_FRAGMENT = re.compile(r"^\d+:\d+:")


def _annotation_value(row):
    """The single segment id a structure's mesh is written under, or None."""
    value = row.get("annotation_value")
    if isinstance(value, list):
        if not value:
            return None
        # write_segment_properties expands a list into several ids, but a mesh can only
        # be emitted under one of them, so mirror the first-value convention used by the
        # precomputed append path and say so rather than dropping the rest silently.
        logging.warning(
            f"Structure {row.get('identifier')!r} has multiple annotation values {value}; "
            f"its mesh will be written under {value[0]} only"
        )
        return value[0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _structure_depth(terms):
    """Depth of each structure, keyed by identifier.

    Uses the precomputed root_identifier_path when the terminology carries one,
    otherwise walks parent_identifier links.
    """
    if "root_identifier_path" in terms.columns:
        return {
            row["identifier"]: len(row["root_identifier_path"]) - 1
            for _, row in terms.iterrows()
        }

    parent_of = dict(zip(terms["identifier"], terms["parent_identifier"]))
    depths = {}

    def depth(identifier):
        if identifier in depths:
            return depths[identifier]
        parent = parent_of.get(identifier)
        depths[identifier] = 0 if parent not in parent_of else depth(parent) + 1
        return depths[identifier]

    return {identifier: depth(identifier) for identifier in terms["identifier"]}


def build_hierarchy_remap_tables(terms):
    """One {leaf annotation value -> structure annotation value} table per depth.

    Each table is unambiguous because structures at the same depth form an antichain
    and so cannot share a descendant.
    """
    if "descendant_annotation_values" not in terms.columns:
        raise ValueError(
            "Terminology does not have 'descendant_annotation_values' column. "
            "Please ensure set_descendant_annotation_values() has been called."
        )

    depth_of = _structure_depth(terms)
    tables = {}

    for _, row in terms.iterrows():
        structure_value = _annotation_value(row)
        if structure_value is None:
            continue

        table = tables.setdefault(depth_of[row["identifier"]], {})
        descendants = row["descendant_annotation_values"]
        for descendant in descendants if descendants is not None else []:
            previous = table.setdefault(int(descendant), int(structure_value))
            if previous != int(structure_value):
                raise ValueError(
                    f"Descendant {descendant} maps to both {previous} and {structure_value} "
                    f"at depth {depth_of[row['identifier']]}; structures at one depth must "
                    f"have disjoint descendants"
                )

    return [tables[depth] for depth in sorted(tables)]


def _clear_mesh_dir(precomputed_output, mesh_dir):
    """Remove a previous run's meshes.

    Stale fragments from structures that have since been removed or renumbered would
    otherwise survive and be re-emitted by the merge pass, which lists whatever it finds.
    """
    path = Path(precomputed_output) / mesh_dir
    if not path.exists():
        return
    removed = 0
    for entry in path.iterdir():
        if entry.is_file():
            entry.unlink()
            removed += 1
    logging.info(f"Cleared {removed} files from {path}")


def _delete_intermediates(precomputed_output, mesh_dir):
    """Drop the raw pass-1 fragments, which are ~20x the size of the final meshes."""
    path = Path(precomputed_output) / mesh_dir
    removed = 0
    for entry in path.iterdir():
        if entry.is_file() and _INTERMEDIATE_FRAGMENT.match(entry.name):
            entry.unlink()
            removed += 1
    logging.info(f"Removed {removed} intermediate mesh fragments from {path}")


def create_multires_meshes(
    precomputed_output,
    terminology,
    mesh_dir="mesh",
    task_shape=(448, 448, 448),
    max_simplification_error=40,
    parallel=8,
    merge_parallel=1,
):
    """Write single-LOD multi-resolution meshes for every structure in the terminology.

    Args:
        precomputed_output: Path to an existing precomputed segmentation layer.
        terminology: Terminology DataFrame with annotation_value and
            descendant_annotation_values columns.
        task_shape: Meshing task size in voxels. Chunk alignment is not required.
        max_simplification_error: Simplification tolerance in nanometres.
        parallel: Worker count for meshing.
        merge_parallel: Worker count for the merge pass. Kept low by default: the merge
            holds an entire structure's mesh in memory, and the root structure spans the
            whole volume, so this pass sets the peak memory of the pipeline.
    """
    # Imported here so the hierarchy helpers above stay usable without igneous installed.
    import igneous.task_creation as tc
    from cloudvolume import CloudVolume
    from igneous.tasks import MeshTask
    from taskqueue import LocalTaskQueue

    precomputed_output = Path(precomputed_output).resolve()
    cloudpath = f"file://{precomputed_output}"

    remap_tables = build_hierarchy_remap_tables(terminology)
    structure_count = sum(len(set(table.values())) for table in remap_tables)
    logging.info(
        f"Meshing {structure_count} structures across {len(remap_tables)} hierarchy levels "
        f"into {precomputed_output / mesh_dir}"
    )

    _clear_mesh_dir(precomputed_output, mesh_dir)

    for depth, table in enumerate(remap_tables):
        structures = len(set(table.values()))
        logging.info(f"Meshing depth {depth}: {structures} structures, {len(table)} source labels")

        tasks = list(
            tc.create_meshing_tasks(
                cloudpath,
                mip=0,
                shape=task_shape,
                mesh_dir=mesh_dir,
                sharded=False,
                # The merge pass reads fragments with Mesh.from_precomputed unconditionally.
                encoding="precomputed",
                # Written per bounding box, so each depth would overwrite the last, leaving
                # an index describing only the final level. The unsharded merge never reads it.
                spatial_index=False,
                # Closes surfaces where a structure is clipped by the volume edge.
                closed_dataset_edges=True,
                # tensorstore omits chunks that are entirely background, which CloudVolume
                # would otherwise report as missing rather than empty.
                fill_missing=True,
                simplification=True,
                max_simplification_error=max_simplification_error,
                compress=None,
            )
        )
        # create_meshing_tasks exposes no remap parameter, but the underlying task accepts
        # one, so the tasks are rebuilt with the level's table attached.
        tasks = [MeshTask(**{**task._args, "remap_table": dict(table)}) for task in tasks]

        queue = LocalTaskQueue(parallel=parallel)
        queue.insert(tasks)
        queue.execute()

    # Must follow every meshing pass: create_meshing_tasks rewrites the mesh info as
    # neuroglancer_legacy_mesh each time it runs, and only the merge restores multilod.
    volume = CloudVolume(cloudpath)
    min_chunk_size = tuple(int(x) for x in volume.meta.bounds(volume.mesh.meta.mip).size3())
    logging.info(f"Merging to multi-resolution meshes, min_chunk_size={min_chunk_size}")

    queue = LocalTaskQueue(parallel=merge_parallel)
    queue.insert(
        tc.create_unsharded_multires_mesh_tasks(
            cloudpath,
            # Forces a single level of detail; see the module docstring.
            num_lod=0,
            magnitude=3,
            mesh_dir=mesh_dir,
            vertex_quantization_bits=16,
            # Bounds every object, so no level of detail is ever split into fragments.
            min_chunk_size=min_chunk_size,
        )
    )
    queue.execute()

    _delete_intermediates(precomputed_output, mesh_dir)
    logging.info(f"Multi-resolution meshes written to {precomputed_output / mesh_dir}")
