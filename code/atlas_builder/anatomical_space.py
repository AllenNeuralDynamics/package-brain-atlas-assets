"""Anatomical coordinate space definition with associated template (moved)."""

from dataclasses import dataclass

from atlas_builder.anatomical_template import AnatomicalTemplate
from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class AnatomicalSpace(AtlasAsset):
    """Anatomical coordinate space with associated template."""

    anatomical_template: AnatomicalTemplate

    _asset_location = "anatomical-spaces"

    @property
    def manifest(self) -> dict:
        return super().manifest | {
            "anatomical_template": self.anatomical_template.manifest,
        }
