"""Complete atlas combining coordinate space, templates, and annotations."""

from dataclasses import dataclass, field
from typing import ClassVar
from pathlib import Path

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.coordinate_space import CoordinateSpace
from atlas_builder.template import Template
from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class Atlas(AtlasAsset):
    """Complete atlas with coordinate space, templates, and annotations."""

    coordinate_space: CoordinateSpace
    annotation_sets: list[AnnotationSet]
    templates: list[Template] = field(default_factory=list)

    _asset_location: ClassVar[str] = "atlases"
    schema_version: ClassVar[str] = "0.1.0"

    @property
    def manifest(self):
        return super().manifest | {
            "coordinate_space": self.coordinate_space.manifest,
            "templates": [t.manifest for t in self.templates],
            "annotation_sets": [a.manifest for a in self.annotation_sets],
        }

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "Atlas":
        coordinate_space = CoordinateSpace.from_manifest(
            manifest["coordinate_space"], root=root
        )
        templates = [
            Template.from_manifest(t, root=root)
            for t in manifest.get("templates", [])
        ]
        annotation_sets = [
            AnnotationSet.from_manifest(a, root=root)
            for a in manifest["annotation_sets"]
        ]
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            coordinate_space=coordinate_space,
            templates=templates,
            annotation_sets=annotation_sets,
        )
