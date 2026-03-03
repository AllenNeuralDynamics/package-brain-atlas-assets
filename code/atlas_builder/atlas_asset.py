"""Base class for atlas assets with location and manifest management (moved from atlas_assets)."""

import json
import logging
from dataclasses import dataclass
from typing import ClassVar, TypeVar
from pathlib import Path


TAtlasAsset = TypeVar("TAtlasAsset", bound="AtlasAsset")


@dataclass
class AtlasAsset:
    """Base class for atlas assets.

    Attributes:
        name: Asset identifier
        version: Asset version string
        _asset_location: Storage directory name for this asset type
        schema_version: class-level schema version (required in subclasses)
    """

    name: str
    version: str

    _asset_location: ClassVar[str] = None
    schema_version: ClassVar[str] = None  # subclasses must override

    @property
    def manifest(self) -> dict:
        """Generate manifest dictionary for this asset.

        Raises:
            ValueError: If subclass did not define schema_version.
        """
        schema_version = getattr(self.__class__, "schema_version", None)
        if not schema_version:
            raise ValueError(f"{self.__class__.__name__} must define schema_version class attribute")
        return {
            "name": self.name,
            "version": self.version,
            "location": str(self.location(Path("/"))),
            "schema_version": schema_version,
        }

    def location(self, root) -> Path:
        """Get file system location for this asset."""
        return Path(root) / self._asset_location / f"{self.name}" / self.version

    @staticmethod
    def resolve_manifest_location(manifest: dict, root: Path | None = None) -> Path:
        """Resolve a manifest location against an optional root path."""
        location = Path(manifest["location"])
        if root is None:
            return location
        if location.is_absolute():
            return Path(root) / location.relative_to("/")
        return Path(root) / location

    @classmethod
    def from_manifest(cls: type[TAtlasAsset], manifest: dict, root: Path | None = None) -> TAtlasAsset:
        """Rehydrate an asset from its manifest."""
        return cls(name=manifest["name"], version=manifest["version"])

    @classmethod
    def from_manifest_file(
        cls: type[TAtlasAsset], manifest_path: Path, root: Path | None = None
    ) -> TAtlasAsset:
        """Rehydrate an asset from a manifest.json file on disk."""
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return cls.from_manifest(manifest, root=root)

    def create_manifest(self, output_root: Path):
        """Create and write manifest.json file for this asset."""
        manifest_file = self.location(output_root) / "manifest.json"
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=3)
        logging.info(f"Created manifest for {self.__class__.__name__}: {self.name} at {manifest_file}")
