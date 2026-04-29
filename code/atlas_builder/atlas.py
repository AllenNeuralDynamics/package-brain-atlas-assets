"""Complete atlas combining coordinate space and annotations."""

from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.coordinate_space import CoordinateSpace
from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class Atlas(AtlasAsset):
    """Complete atlas with coordinate space and annotations."""

    coordinate_space: CoordinateSpace
    annotation_sets: list[AnnotationSet]

    _asset_location: ClassVar[str] = "atlases"
    schema_version: ClassVar[str] = "0.1.0"

    @property
    def manifest(self):
        return super().manifest | {
            "coordinate_space": self.coordinate_space.manifest,
            "annotation_sets": [a.manifest for a in self.annotation_sets],
        }

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "Atlas":
        coordinate_space = CoordinateSpace.from_manifest(
            manifest["coordinate_space"], root=root
        )
        annotation_sets = [
            AnnotationSet.from_manifest(a, root=root)
            for a in manifest["annotation_sets"]
        ]
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            coordinate_space=coordinate_space,
            annotation_sets=annotation_sets,
        )
