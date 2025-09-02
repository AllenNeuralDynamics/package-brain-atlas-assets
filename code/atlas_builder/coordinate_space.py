"""Coordinate space definition with associated template."""

from dataclasses import dataclass

from atlas_builder.template import Template
from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class CoordinateSpace(AtlasAsset):
    """Coordinate space with associated template."""

    template: Template

    _asset_location = "coordinate-spaces"

    @property
    def manifest(self) -> dict:
        return super().manifest | {
            "template": self.template.manifest,
        }
