"""Coordinate space definition with associated template."""

from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

from atlas_builder.template import Template
from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class CoordinateSpace(AtlasAsset):
    """Coordinate space with associated template."""

    template: Template

    _asset_location: ClassVar[str] = "coordinate-spaces"
    schema_version: ClassVar[str] = "0.1.0"

    @property
    def manifest(self) -> dict:
        return super().manifest | {
            "template": self.template.manifest,
        }

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "CoordinateSpace":
        template = Template.from_manifest(manifest["template"], root=root)
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            template=template,
        )
