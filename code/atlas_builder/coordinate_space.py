"""Coordinate space definition."""

from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class CoordinateSpace(AtlasAsset):
    """Coordinate space."""

    _asset_location: ClassVar[str] = "coordinate-spaces"
    schema_version: ClassVar[str] = "0.1.0"

    @classmethod
    def from_manifest(cls, manifest: dict, root: Path | None = None) -> "CoordinateSpace":
        return cls(
            name=manifest["name"],
            version=manifest["version"],
        )
