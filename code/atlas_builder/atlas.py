"""Complete atlas combining coordinate space, annotations, and terminology."""

from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.coordinate_space import CoordinateSpace
from atlas_builder.atlas_asset import AtlasAsset
from atlas_builder.terminology import Terminology


@dataclass
class Atlas(AtlasAsset):
    """Complete atlas with coordinate space, annotations, and terminology."""

    coordinate_space: CoordinateSpace
    annotation_set: AnnotationSet
    terminology: Terminology

    _asset_location: ClassVar[str] = "atlases"
    schema_version: ClassVar[str] = "0.1.0"

    @property
    def manifest(self):
        return super().manifest | {
            "coordinate_space": self.coordinate_space.manifest,
            "annotation_set": self.annotation_set.manifest,
            "terminology": self.terminology.manifest,
        }

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "Atlas":
        coordinate_space = CoordinateSpace.from_manifest(
            manifest["coordinate_space"], root=root
        )
        annotation_set = AnnotationSet.from_manifest(
            manifest["annotation_set"], root=root
        )
        terminology = Terminology.from_manifest(manifest["terminology"], root=root)
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            coordinate_space=coordinate_space,
            annotation_set=annotation_set,
            terminology=terminology,
        )
