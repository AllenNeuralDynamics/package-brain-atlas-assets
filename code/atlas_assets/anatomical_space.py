"""Anatomical coordinate space definition with associated template."""

from dataclasses import dataclass

from atlas_assets.anatomical_template import AnatomicalTemplate
from atlas_assets.atlas_asset import AtlasAsset


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
